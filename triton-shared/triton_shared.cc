// LLVM
#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"

// MLIR: Conversion Passes
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"

// MLIR: Dialects
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
// #include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"

// MLIR: Core IR and Passes
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// MLIR: Target and Translation
// #include "mlir/Target/LLVMIR/Dialect/AMX/AMXToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

// LLVM: Debug
#include "llvm/Support/Debug.h" // Key header file

// MLIR: Top-level Transforms
#include "mlir/Transforms/Passes.h"

// Triton and other third-party dialects
#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#define ADD_PASS_WRAPPER_0(name, builder)                                      \
  m.def(name, [](mlir::PassManager &pm) { pm.addPass(builder()); })

#define ADD_PASS_WRAPPER_1(name, builder, ty0)                                 \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder(val0)); })

#define ADD_PASS_WRAPPER_1_ARG(name, builder, ty0, arg0, val0)                 \
  m.def(                                                                       \
      name,                                                                    \
      [](mlir::PassManager &pm, ty0 arg0) { pm.addPass(builder(val0)); },      \
      py::arg("pm"), py::arg(#arg0) = val0);

void init_to_llvm(py::module &&m) {
  using namespace mlir;

  ADD_PASS_WRAPPER_0("add_transform_interpreter",
                     transform::createInterpreterPass);

  ADD_PASS_WRAPPER_0("add_convert_linalg_to_affine_loops",
                     createConvertLinalgToAffineLoopsPass);
  ADD_PASS_WRAPPER_0("add_empty_tensor_to_alloc_tensor",
                     bufferization::createEmptyTensorToAllocTensorPass);

  ADD_PASS_WRAPPER_1_ARG(
    "add_convert_vector_to_llvm_with_sve",
    [](bool enableSVE) {
        mlir::ConvertVectorToLLVMPassOptions options;
        options.armSVE = enableSVE;
        return mlir::createConvertVectorToLLVMPass(options);
    },
    bool, enable_sve, true);


  ADD_PASS_WRAPPER_0("add_canonicalize", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_one_shot_bufferize",
                     bufferization::createOneShotBufferizePass);
  ADD_PASS_WRAPPER_0("add_lower_affine", createLowerAffinePass);
  ADD_PASS_WRAPPER_0("add_convert_linalg_to_loops",
                     createConvertLinalgToLoopsPass);
  ADD_PASS_WRAPPER_0("add_expand_strided_metadata",
                     memref::createExpandStridedMetadataPass);
  ADD_PASS_WRAPPER_0("add_convert_scf_to_cf", createConvertSCFToCFPass);
  ADD_PASS_WRAPPER_0("add_convert_arith_to_llvm",
                     createArithToLLVMConversionPass);
  ADD_PASS_WRAPPER_0("add_convert_math_to_llvm", createConvertMathToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_complex_to_llvm",
                     createConvertComplexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_vector_to_llvm",
                     createConvertVectorToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_index_to_llvm", createConvertIndexToLLVMPass);
  ADD_PASS_WRAPPER_0("add_memref_expand", memref::createExpandOpsPass);
  ADD_PASS_WRAPPER_0("add_finalize_memref_to_llvm",
                     createFinalizeMemRefToLLVMConversionPass);
  ADD_PASS_WRAPPER_0("add_convert_func_to_llvm", createConvertFuncToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_tptr_to_llvm", tptr::createTPtrToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_cf_to_llvm",
                     createConvertControlFlowToLLVMPass);
  ADD_PASS_WRAPPER_0("add_reconcile_unrealized_casts",
                     createReconcileUnrealizedCastsPass);
}

void init_triton_triton_shared(py::module &&m) {

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;

    // Register core dialects
    registry.insert<
        ::mlir::triton::TritonDialect, ::mlir::linalg::LinalgDialect,
        ::mlir::bufferization::BufferizationDialect, ::mlir::tptr::TPtrDialect,
        ::mlir::math::MathDialect, ::mlir::memref::MemRefDialect,
        ::mlir::arith::ArithDialect, ::mlir::scf::SCFDialect,
        ::mlir::vector::VectorDialect, ::mlir::cf::ControlFlowDialect,
        ::mlir::LLVM::LLVMDialect, ::mlir::ub::UBDialect,
        ::mlir::func::FuncDialect, ::mlir::transform::TransformDialect>();

    // Register interfaces and translations
    // ::mlir::registerAllDialects(registry);
    ::mlir::registerBuiltinDialectTranslation(registry);
    ::mlir::registerLLVMDialectTranslation(registry);

    // Register bufferizable op interface external models
    ::mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    ::mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    ::mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    ::mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
    ::mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);

    ::mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    // ::mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);

    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
