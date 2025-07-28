from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, cpu, llvm
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
from mlir.dialects.transform import structured, loop, vector, bufferization, tensor


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
        self.cpu_features = llvm.get_cpu_features()
        self.cpu_arch = llvm.get_cpu_tripple().split("-")[0]

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

    def _ttir_to_ttsharedir(self, mod):
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



    def _sve_transform(self, src: str) -> str:
        ## Transform needed for SVE (as seen in the official MLIR example)
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

        ## instead of using mlir-opt we embed the optimization passes in the transform dialect 
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
                    # optimize
                    transform.IncludeOp(
                        [],
                        FlatSymbolRefAttr.get("opt"),
                        transform.FailurePropagationMode.Propagate,
                        [foreachBody.arguments[0]],
                    )

                    transform.YieldOp([])

                transform.YieldOp([])
                     
        with Context() as ctx, Location.unknown():
            mod = Module.create()
            ## add attributes to the module
            mod.operation.attributes["transform.with_named_sequence"] = UnitAttr.get()
            
            with InsertionPoint(mod.body):
                tileAndVectorize()
                opt()
                transform_main()

            ## Append our transform to the original source
            return src + "\n" + str(mod)


    def _sme_transform(self, src: str) -> str:

        def step1():
            sequence = transform.NamedSequenceOp(
                "__step1",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}],
            )
                
            with InsertionPoint(sequence.body):

                ## get matmul
                matmuls = structured.MatchOp.match_op_names(
                    sequence.bodyTarget,
                    ["linalg.matmul"]
                )
                
                # Step 1: Tile for size [4] x [4], which corresponds to SVLs x SVLs, where
                # SVLs is the number of 32-bit elements in a vector of SVL bits.
                tiled = structured.TileUsingForOp(matmuls.result, sizes=[[4],[4],1])
                # Step 2: Vectorize.
                structured.VectorizeOp(tiled.results[0], [[4], [4], 1])
                # Step 3: Bufferize ahead of TransferReadDropUnitDimsPattern, which
                # currently only supports memrefs.
                buff = bufferization.OneShotBufferizeOp(
                    sequence.bodyTarget, bufferize_function_boundaries=True)
                ## get the funcs
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    buff.result,
                    ["func.func"]
                )

                l = transform.ApplyRegisteredPassOp(
                    transform.OperationType.get("func.func"),
                    funcs.result,
                    "convert-linalg-to-loops",
                )

                # Step 4: Lower vector.multi_reduction to vector.contract (+ some helpful patterns).
                with InsertionPoint(transform.ApplyPatternsOp(l).patterns):
                    vector.ApplyLowerMaskedTransfersPatternsOp()
                    vector.ApplyTransferPermutationPatternsOp()
                    vector.ApplyVectorReductionToContractPatternsOp()

                # Step 5: Lower vector.contract to vector.outerproduct. Also drop unit
                # dims, specifically to prevent vector.transfer_read of vector<[4]x1xf32>,
                # which can't be lowered in generic path.
                with InsertionPoint(transform.ApplyPatternsOp(l).patterns):
                    vector.ApplyCastAwayVectorLeadingOneDimPatternsOp()
                    tensor.ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp()
                    vector.ApplyLowerContractionPatternsOp(lowering_strategy=vector.VectorContractLowering.OuterProduct)
                    vector.ApplyLowerMasksPatternsOp()
                    transform.ApplyCanonicalizationPatternsOp()

                all_loops = structured.MatchOp.__base__(
                    transform.AnyOpType.get(),
                    buff.result,
                    interface=structured.MatchInterfaceEnum.LoopLikeInterface
                )

                transform.apply_licm(
                    all_loops.result,
                )

                loop.loop_hoist_loop_invariant_subsets(
                    all_loops.result,
                )

                transform.YieldOp([buff.result])
 
        def arm_sme_lowering_schedule():
            sequence = transform.NamedSequenceOp(
                "__arm_sme_lowering_schedule",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )
            with InsertionPoint(sequence.body):
                result = transform.lower_to_arm_sme(
                    sequence.bodyTarget,
                )
                transform.YieldOp([sequence.bodyTarget])

        def lower_to_llvm():
            sequence = transform.NamedSequenceOp(
                "__lower_to_llvm",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )
            with InsertionPoint(sequence.body):
                result = transform.lower_to_llvm_new(
                    sequence.bodyTarget,
                    enable_arm_sve=True,
                    enable_index_optimizations=True,
                    vscale_range=0,
                )
                transform.YieldOp([sequence.bodyTarget])
                    
        def step2(include, name):
            sequence = transform.NamedSequenceOp(
                "__step2_" + name,
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}],
            )
            with InsertionPoint(sequence.body):
                ## get all funcs
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    sequence.bodyTarget,
                    ["func.func"]
                )
                ## get parent op
                p = transform.get_parent_op(
                    transform.AnyOpType.get(),
                    funcs.result, 
                    deduplicate=True,
                )
                ## include
                sme = transform.IncludeOp(
                    [transform.AnyOpType.get()],
                    include,
                    transform.FailurePropagationMode.Propagate,
                    [p],
                )
                
                ## cse
                cse = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    sme.result,
                    "cse",
                )

                with InsertionPoint(transform.ApplyPatternsOp(cse).patterns):
                    structured.apply_patterns_linalg_tiling_canonicalization()
                    loop.apply_patterns_scf_for_loop_canonicalization()
                
                ## match looplike
                looplike = structured.MatchOp.__base__(
                    transform.AnyOpType.get(),
                    cse.result,
                    interface=structured.MatchInterfaceEnum.LoopLikeInterface
                )
                ## apply licm
                transform.apply_licm(
                    looplike.result,
                )
                ## match func from cse
                funcs = structured.MatchOp.match_op_names(
                    transform.AnyOpType.get(),
                    cse.result,
                    ["func.func"]
                )
                ## hoist redudant vector transfers
                a = transform.structured.HoistRedundantVectorTransfersOp(
                    transform.AnyOpType.get(),
                    funcs.result,
                )
                ## hoist redundant vector broadcasts
                b = transform.structured.HoistRedundantVectorBroadcastsOp(
                    transform.AnyOpType.get(),
                    a.result,
                )
                ## canonicalize
                transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    b.result,
                    "canonicalize",
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
                ## include step 1
                first = transform.IncludeOp(
                    [transform.AnyOpType.get()],
                    FlatSymbolRefAttr.get("__step1"),
                    transform.FailurePropagationMode.Propagate,
                    [sequence.bodyTarget],
                )
                ## include step 2 sme
                transform.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get("__step2_sme"),
                    transform.FailurePropagationMode.Propagate,
                    [first],
                )

                ## include step 2 llvm
                transform.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get("__step2_llvm"),
                    transform.FailurePropagationMode.Propagate,
                    [first],
                )

                transform.YieldOp([])
        
        
        with Context() as ctx, Location.unknown():
            mod = Module.create()
            ## add attributes to the module
            mod.operation.attributes["transform.with_named_sequence"] = UnitAttr.get()
            
            with InsertionPoint(mod.body):
                step1()
                arm_sme_lowering_schedule()
                lower_to_llvm()
                step2(FlatSymbolRefAttr.get("__arm_sme_lowering_schedule"), "sme")
                step2(FlatSymbolRefAttr.get("__lower_to_llvm"), "llvm")
                transform_main()

            ## Append our transform to the original source
            return src + "\n" + str(mod)

       

    def _optimize_ttsharedir(self, src: str):
        if True:
        #if(self.cpu_arch == "aarch64" and "sme" in self.cpu_features):
            return self._sme_transform(src)
        elif (self.cpu_arch == "aarch64" and "sve" in self.cpu_features):
            return self._sve_transform(src)


        return src

    def _extract_mlir_function(self, filepath: str) -> None:
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
                if decl_pattern.match(line):
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
            return

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


    def _ttsharedir_to_llir(self, ttsharedir: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
            llmlir_path = os.path.join(tmpdir, "ll.mlir")
            llir_path = os.path.join(tmpdir, "ll.ir")
            Path(ttshared_path).write_text(ttsharedir)
            mlir_opt_path = _get_llvm_bin_path("mlir-opt")

            if True:
            #if self.cpu_arch == "aarch64" and "sme" in self.cpu_features:
                pipeline = [
                "--transform-interpreter",
                "--test-transform-dialect-erase-schedule",
                ]
            elif self.cpu_arch == "aarch64" and "sve" in self.cpu_features:
                pipeline = [
                "--transform-interpreter",
                "--test-transform-dialect-erase-schedule",
                "--convert-vector-to-llvm=\"enable-arm-sve\"",
                "--convert-linalg-to-loops",
                "--test-lower-to-llvm",
                ]
            else:
                pipeline = [
                "--convert-linalg-to-affine-loops",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                "--lower-affine",
                "--convert-linalg-to-loops",
                "--expand-strided-metadata",
                "--convert-scf-to-cf",
                "--test-lower-to-llvm",
                "--reconcile-unrealized-casts",
                ]
           
            _dump_ir_if_needed([ttshared_path])
            # TritonShared-MLIR to LLVM-MLIR
            subprocess.check_call([mlir_opt_path, ttshared_path] + pipeline + [ "-o", llmlir_path])

            _dump_ir_if_needed([llmlir_path])
            self._extract_mlir_function(llmlir_path)
            # LLVM-MLIR to LLVM-IR
            mlir_translate_path = _get_llvm_bin_path("mlir-translate")
            subprocess.check_call([mlir_translate_path, llmlir_path,
                "--mlir-to-llvmir",
                "-o",
                llir_path])
            _dump_ir_if_needed([llir_path])
            return Path(llir_path).read_text()


    def _optimize_llir(self, llir: str):
        # We don't apply any optimizations now, but we can add passes if needed.
        return llir


    def _llir_to_bin(self, llir: str, metadata):
        pattern = r"define void @(\w+)\(.+"
        matches = re.findall(pattern, llir)
        assert len(matches) == 1
        metadata["name"] = matches[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "kernel.ll")
            dst_path = os.path.join(tmpdir, "kernel.o")
            Path(src_path).write_text(llir)
            llc_path = _get_llvm_bin_path("llc")
            flags = ""
            if self.cpu_arch == "aarch64" and "sme" in self.cpu_features:
                flags = (
                    "-mtriple=aarch64-linux-gnu",
                    "-mattr=+sme",
                )
            elif self.cpu_arch == "aarch64" and "sve" in self.cpu_features:
                flags = (
                    "-mtriple=aarch64-linux-gnu",
                    "-mattr=+sve",
                )
            
            subprocess.check_call([llc_path, src_path, "-filetype=obj", "-o", dst_path] + list(flags))
            return Path(dst_path).read_bytes()



    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: self._optimize_ttsharedir(self._ttir_to_ttsharedir(src))
        stages["llir"] = lambda src, metadata: self._optimize_llir(self._ttsharedir_to_llir(src))
        stages["obj"] = lambda src, metadata: self._llir_to_bin(src, metadata)


    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
