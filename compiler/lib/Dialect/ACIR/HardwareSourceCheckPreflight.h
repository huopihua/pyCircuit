#ifndef PYCIRCUIT_HARDWARE_SOURCE_CHECK_PREFLIGHT_H
#define PYCIRCUIT_HARDWARE_SOURCE_CHECK_PREFLIGHT_H

#include "HardwareSourceChecks.h"
#include "PackageVerification.h"
#include "SourceCheckBudget.h"
#include "pycircuit/Analysis/CompilerPhaseMetrics.h"

namespace acir::ac::detail {
/// Invocation-local measurement for capacity and scaling regressions. Counts
/// describe the completed or partially constructed plan; they never participate
/// in admission or become semantic authority.
struct SourceCheckBuildMetrics {
  uint64_t workUnits = 0;
  uint64_t definitions = 0;
  uint64_t boundDefinitions = 0;
  uint64_t preparedSampledContexts = 0;
  uint64_t sampledRuleParses = 0;
  uint64_t sampledAdmissionProofs = 0;
  uint64_t ownerOccurrences = 0;
  uint64_t resolvedChecks = 0;
  uint64_t calls = 0;
  uint64_t instances = 0;
  uint64_t irWalks = 0;
  uint64_t visitedNodes = 0;
  uint64_t planEntries = 0;
  uint64_t planBytes = 0;
};

/// Share the plan builder's immutable native-verification invocation with
/// ModuleOp's parameterless ODS verifier. The scope is synchronous and bound to
/// one exact package; nested scopes restore the prior context on every exit.
/// Standalone Module verification has no scope and keeps its independent proof.
class NativeSourceCheckScope {
public:
  using Definitions = llvm::DenseMap<mlir::Operation *,
                                    llvm::SmallVector<HardwareCheckBinding>>;
  NativeSourceCheckScope(const HardwareAnalysis &analysis,
                        SampledAdmissionValidation &validation,
                        SourceCheckBudget &budget, Definitions &definitions);
  ~NativeSourceCheckScope();
  NativeSourceCheckScope(const NativeSourceCheckScope &) = delete;
  NativeSourceCheckScope &operator=(const NativeSourceCheckScope &) = delete;
  static std::optional<mlir::LogicalResult> verify(ModuleOp definition);
private:
  static thread_local NativeSourceCheckScope *active;
  NativeSourceCheckScope *previous;
  const HardwareAnalysis &analysis;
  SampledAdmissionValidation &validation;
  SourceCheckBudget &budget;
  Definitions &definitions;
};
mlir::FailureOr<HardwareSourceCheckPlan> buildSourceCheckPlan(
    const HardwareAnalysis &analysis, HardwareSourceCheckLimits limits,
    const PackageStagePolicy &policy = PackageStagePolicy::hardware(),
    SourceCheckBuildMetrics *metrics = nullptr);
void publishSourceCheckMetrics(const SourceCheckBuildMetrics &source,
                               acir::metrics::CompilerPhaseMetrics &target);
} // namespace acir::ac::detail
#endif
