#include "pycircuit/Dialect/ACIR/GFIRPackageValidation.h"
#include "HardwareSourceCheckPreflight.h"
#include "pycircuit/Dialect/ACIR/HardwareAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace acir::ac {
LogicalResult verifyRoutedPackage(mlir::ModuleOp package,
                                  HardwareSourceCheckLimits limits,
                                  acir::metrics::CompilerPhaseMetrics *metrics) {
  if (!package)
    return failure();
  HardwareAnalysis analysis(package);
  detail::SourceCheckBuildMetrics buildMetrics;
  auto plan = detail::buildSourceCheckPlan(
      analysis, limits, detail::PackageStagePolicy::routed(),
      metrics ? &buildMetrics : nullptr);
  if (metrics)
    detail::publishSourceCheckMetrics(buildMetrics, *metrics);
  return success(succeeded(plan));
}
} // namespace acir::ac

namespace acir {
namespace {

class VerifyRoutedPass
    : public PassWrapper<VerifyRoutedPass, OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyRoutedPass)
  StringRef getArgument() const final { return "ac-verify-routed"; }
  StringRef getDescription() const final {
    return "Verify the routed logical-state ACIR anchor without source objects";
  }
  void runOnOperation() final {
    if (failed(ac::verifyRoutedPackage(getOperation())))
      signalPassFailure();
    else
      markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createVerifyRoutedPass() {
  return std::make_unique<VerifyRoutedPass>();
}

void registerGFIRValidationPasses() {
  PassRegistration<VerifyRoutedPass>();
}
} // namespace acir
