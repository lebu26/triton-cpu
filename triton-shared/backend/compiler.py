from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import shutil
import subprocess
import functools
import textwrap
from pathlib import Path
from pdb import set_trace as st
from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import pdl as transform_pdl
from mlir.dialects.transform import structured, loop, vector, bufferization

def _get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def _dump_ir_if_needed(files):
    path = os.getenv("TRITON_SHARED_DUMP_PATH", "")
    if not path:
        return
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))


def _ttir_to_ttsharedir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        _dump_ir_if_needed([src_path])
        triton_shared_opt_path = _get_triton_shared_opt_path()
        subprocess.check_call([triton_shared_opt_path, src_path, "--triton-to-linalg-experimental", "-o", dst_path])
        return Path(dst_path).read_text()


def _optimize_ttsharedir(ttsharedir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttsharedir

def extract_mlir_function(filepath: str) -> None:
    """
    Reads an MLIR source file, retains external llvm.func declarations and
    extracts the first llvm.func definition with a body, then overwrites the file
    with these declarations followed by the function body.

    Args:
        filepath: Path to the MLIR source file to process.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    decl_pattern = re.compile(r"^\s*llvm\.func\b.*\)(?:\s*->.*)?\s*$")
    body_start_pattern = re.compile(r"^\s*llvm\.func\b.*\{")

    decl_lines = []
    body_lines = []
    in_body = False
    brace_balance = 0

    for line in lines:
        if not in_body:
            # Collect external declarations (no body)
            if decl_pattern.match(line) and not body_start_pattern.search(line):
                decl_lines.append(line)
                continue
            # Detect start of first function with body
            if body_start_pattern.search(line):
                in_body = True
                brace_balance = line.count('{') - line.count('}')
                body_lines.append(line)
                continue
        else:
            body_lines.append(line)
            brace_balance += line.count('{') - line.count('}')
            if brace_balance == 0:
                break

    if not body_lines:
        raise ValueError(f"No 'llvm.func' definition with body found in {filepath}")

    # Dedent the function body block
    dedented_body = textwrap.dedent(''.join(body_lines)).strip() + '\n'

    # Combine declarations and body
    output = ''.join(decl_lines).strip()
    if decl_lines:
        output += '\n\n'
    output += dedented_body

    # Overwrite file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(output)


def _ttsharedir_to_llir(ttsharedir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        mlir_opt_path = _get_llvm_bin_path("mlir-opt")

        def tileAndVectorize():
            sequence = transform.NamedSequenceOp(
                "__tile_and_vectorize",
                [transform.OperationType.get("func.func")],
                [],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )
            with InsertionPoint(sequence.body):
                # Step 0: Get a handle to the matmul op
                matmuls = structured.MatchOp.match_op_names(
                    sequence.bodyTarget,
                    ["linalg.matmul"] 
                )
                
                # Step 1: Tile
                tiled = structured.TileUsingForOp(matmuls.result, sizes=[2,[4],1])

                # Step 2: Vectorize
                structured.VectorizeOp(tiled.results[0], [2, [4], 1])
                
                # Step 3: Lower vector.multi_reduction to vector.contract (+ some helpful patterns)
                with InsertionPoint(transform.ApplyPatternsOp(sequence.bodyTarget).patterns):
                    vector.ApplyVectorReductionToContractPatternsOp()
                    vector.ApplyTransferPermutationPatternsOp()
                    vector.ApplyLowerMaskedTransfersPatternsOp()
                    # vector.ApplySinkVectorPatternsOp() # not available in LLVM 19

                with InsertionPoint(transform.ApplyPatternsOp(sequence.bodyTarget).patterns):
                    vector.ApplyLowerContractionPatternsOp(lowering_strategy=vector.VectorContractLowering.OuterProduct)
                    vector.ApplyLowerOuterProductPatternsOp()
                    
                transform.YieldOp([])

        def opt():
            sequence = transform.NamedSequenceOp(
                "opt",
                [transform.OperationType.get("func.func")],
                [],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}],
            )

            with InsertionPoint(sequence.body):

                transform.apply_cse(
                    sequence.bodyTarget,)

                c = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    sequence.bodyTarget,
                    "canonicalize",)
                
                s = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    c.result,
                    "convert-vector-to-scf",)
                
                l = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    s.result,
                    "convert-linalg-to-loops",
                )
                
                sve = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    l.result,
                    "arm-sve-legalize-vector-storage",
                )

                transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    sve.result,
                    "convert-vector-to-llvm",
                    options='enable-arm-sve',
                )
                
                transform.YieldOp([])

        def transform_main():
            sequence = transform.NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}],
            )
            
            with InsertionPoint(sequence.body):
                
                buff = bufferization.OneShotBufferizeOp(sequence.bodyTarget, bufferize_function_boundaries= True)
                
                # get all the functions
                funcs = structured.MatchOp.match_op_names(
                    transform.OperationType.get("func.func"),
                    buff.result,
                    ["func.func"]
                )
                ## for each
                foreach = transform.ForeachOp(
                    [],
                    funcs,
                )
                
                foreachBody = foreach.body.blocks.append(transform.OperationType.get("func.func"))
                
                with InsertionPoint(foreachBody):
                    # tile and vectorize
                    x = transform.IncludeOp(
                        [],
                        FlatSymbolRefAttr.get("__tile_and_vectorize"),
                        transform.FailurePropagationMode.Propagate,
                        [foreachBody.arguments[0]],
                    )

                    transform.IncludeOp(
                        [],
                        FlatSymbolRefAttr.get("opt"),
                        transform.FailurePropagationMode.Propagate,
                        [foreachBody.arguments[0]],
                    )

                    transform.YieldOp([])

                transform.YieldOp([])
                 
        with Context() as ctx, Location.file (ttshared_path, line=0, col=0, context=ctx):
            ctx.allow_unregistered_dialects = True
            ctx.load_all_available_dialects()
            ctx.enable_multithreading = False
            mod = Module.create()
            ## add attributes to the module
            mod.operation.attributes["transform.with_named_sequence"] = UnitAttr.get()
        
            with InsertionPoint(mod.body):
                tileAndVectorize()
                opt()
                transform_main()

            ## write mod back to file
            with open(ttshared_path, 'a') as f:
                print(mod, file=f)
        
        _dump_ir_if_needed([ttshared_path])
        # TritonShared-MLIR to LLVM-MLIR
        subprocess.check_call([mlir_opt_path, ttshared_path,
            "--transform-interpreter",
            "--test-transform-dialect-erase-schedule",
            '--convert-vector-to-llvm="enable-arm-sve"',
            "--convert-linalg-to-loops",
            "--test-lower-to-llvm",
            "-o",
            llmlir_path])

        extract_mlir_function(llmlir_path)
        _dump_ir_if_needed([llmlir_path])
        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_path,
            "--mlir-to-llvmir",
            "-o",
            llir_path])
        _dump_ir_if_needed([llir_path])
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return llir


def _llir_to_bin(llir: str, metadata):
    pattern = r"define void @(\w+)\(.+"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1
    metadata["name"] = matches[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)
        llc_path = _get_llvm_bin_path("llc")
        subprocess.check_call([llc_path, src_path, "-filetype=obj", "-o", dst_path])
        return Path(dst_path).read_bytes()



@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_threads: int = 1
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = 'obj'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cpu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        return CPUOptions(**args)

    def get_codegen_implementation(self):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(_ttir_to_ttsharedir(src))
        stages["llir"] = lambda src, metadata: _optimize_llir(_ttsharedir_to_llir(src))
        stages["obj"] = lambda src, metadata: _llir_to_bin(src, metadata)


    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
