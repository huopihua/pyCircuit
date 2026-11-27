#ifndef PYCIRCUIT_CONVERSION_GFIRTOHWIR_H
#define PYCIRCUIT_CONVERSION_GFIRTOHWIR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include "pycircuit/Analysis/CompilerPhaseMetrics.h"
#include <memory>

namespace mlir {
class Pass;
}

namespace acir::ac::detail {
class SourceCheckBudget;
}

namespace acir::detail {
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> materializeHardwareState(
    mlir::ModuleOp routed, ac::detail::SourceCheckBudget &budget);
mlir::LogicalResult verifyHardwareStateMaterialization(
    mlir::ModuleOp routed, mlir::ModuleOp candidate,
    ac::detail::SourceCheckBudget &budget,
    acir::metrics::CompilerPhaseMetrics *metrics = nullptr);
} // namespace acir::detail

namespace acir {
std::unique_ptr<mlir::Pass> createLowerTransactionsPass(
    acir::metrics::CompilerPhaseMetrics *finalVerification = nullptr);
void registerGFIRToHWIRPasses();
} // namespace acir
#endif
