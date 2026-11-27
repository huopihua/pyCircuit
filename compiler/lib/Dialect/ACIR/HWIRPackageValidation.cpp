#include "pycircuit/Dialect/ACIR/HWIRPackageValidation.h"
#include "HardwareSourceCheckPreflight.h"
#include "pycircuit/Dialect/ACIR/HardwareAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace acir::ac {
LogicalResult verifyHardwarePackage(
    mlir::ModuleOp package,
    acir::metrics::CompilerPhaseMetrics *metrics) {
  if (!package)
    return failure();
  HardwareAnalysis analysis(package);
  detail::SourceCheckBuildMetrics buildMetrics;
  auto plan = detail::buildSourceCheckPlan(
      analysis, {}, detail::PackageStagePolicy::hardware(),
      metrics ? &buildMetrics : nullptr);
  if (metrics)
    detail::publishSourceCheckMetrics(buildMetrics, *metrics);
  return success(succeeded(plan));
}
} // namespace acir::ac

namespace acir {
namespace {

class VerifyHardwarePass
    : public PassWrapper<VerifyHardwarePass, OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyHardwarePass)
  StringRef getArgument() const final { return "ac-verify-hardware"; }
  StringRef getDescription() const final {
    return "Verify the closed module/rule/instance/bits hardware graph";
  }
  void runOnOperation() final {
    if (failed(getAnalysis<ac::HardwareAnalysis>().verify()))
      signalPassFailure();
    else
      markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createVerifyHardwarePass() {
  return std::make_unique<VerifyHardwarePass>();
}

void registerHWIRValidationPasses() {
  PassRegistration<VerifyHardwarePass>();
}
} // namespace acir
