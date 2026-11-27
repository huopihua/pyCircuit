#ifndef PYCIRCUIT_DIALECT_ACIR_HWIRPACKAGEVALIDATION_H
#define PYCIRCUIT_DIALECT_ACIR_HWIRPACKAGEVALIDATION_H

#include "pycircuit/Dialect/ACIR/HardwareAnalysisContext.h"
#include "pycircuit/Analysis/CompilerPhaseMetrics.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace mlir {
class Pass;
}

namespace acir::ac {
/// Verify common hardware shape, binding, dependency and execution contracts.
mlir::LogicalResult verifyHardwarePackage(
    mlir::ModuleOp package,
    acir::metrics::CompilerPhaseMetrics *metrics = nullptr);
} // namespace acir::ac

namespace acir {
std::unique_ptr<mlir::Pass> createVerifyHardwarePass();
/// Register the ac-verify-hardware pass from the HWIR validation target.
void registerHWIRValidationPasses();
} // namespace acir
#endif
