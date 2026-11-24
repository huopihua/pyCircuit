#include "Compiler/BackendRenderPlan.h"
#include "Compiler/HardwareEmissionPlan.h"
#include "Compiler/SourceUnit.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "pycircuit/Dialect/ACIR/ACIRDialect.h"
#include "pycircuit/Dialect/GFIR/GFIRDialect.h"

#include <type_traits>

static_assert(!std::is_copy_constructible_v<
              acir::compiler::VerifiedHardwareEmissionPlan>);
static_assert(
    !std::is_copy_constructible_v<acir::compiler::VerifiedCppRenderPlan>);
static_assert(
    !std::is_copy_constructible_v<acir::compiler::VerifiedVerilogRenderPlan>);

int main() {
  mlir::DialectRegistry registry;
  registry.insert<acir::ac::ACIRDialect, acir::gfir::GFIRDialect>();
  mlir::MLIRContext context(registry);
  auto *dialect = context.getOrLoadDialect<acir::ac::ACIRDialect>();

  // Exercise an exported CompilerDev symbol so the consumer proves that the
  // source compiler library and its public header closure link together.
  auto diagnostic = [&]() {
    return mlir::emitError(mlir::UnknownLoc::get(&context));
  };
  llvm::SmallVector<mlir::ModuleOp> headers;
  auto sourceHeaders = acir::compiler::SourceHeaderRegistry::create(
      &context, headers, diagnostic);

  acir::compiler::SourceUnitArtifacts unitArtifacts;
  acir::compiler::CppRenderPlan cppRenderPlan;
  acir::compiler::VerilogRenderPlan verilogRenderPlan;
  acir::compiler::HardwareEmissionPlan hardwarePlan;
  return dialect == nullptr || mlir::failed(sourceHeaders) ||
                 mlir::succeeded(acir::compiler::verifyHardwareEmissionPlan(
                     hardwarePlan)) ||
                 !cppRenderPlan.parts.sourceGroups.empty() ||
                 !verilogRenderPlan.parts.sourceGroups.empty() ||
                 unitArtifacts.body
             ? 1
             : 0;
}
