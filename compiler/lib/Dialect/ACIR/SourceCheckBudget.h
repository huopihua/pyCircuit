#ifndef PYCIRCUIT_SOURCE_CHECK_BUDGET_H
#define PYCIRCUIT_SOURCE_CHECK_BUDGET_H

#include "pycircuit/Dialect/ACIR/HardwareAnalysisContext.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"
#include "pycircuit/Analysis/CompilerPhaseMetrics.h"
#include <algorithm>
#include <bit>
#include <limits>

namespace acir::ac::detail {
/// Invocation-local work accounting. Saturation never permits wraparound to
/// turn an unaffordable closure into an admitted one.
class SourceCheckBudget {
public:
  explicit SourceCheckBudget(HardwareSourceCheckLimits limits)
      : limit(limits.maxWorkUnits) {}
  /// Bound one independently replayable item (for example, one frozen
  /// transaction invocation) without charging its work again to every parent
  /// item. The enclosing scope resumes at its prior usage; peak work and
  /// diagnostic totals still include the nested scope.
  class LocalScope {
  public:
    explicit LocalScope(SourceCheckBudget &budget)
        : LocalScope(&budget) {}
    explicit LocalScope(SourceCheckBudget *budget)
        : budget(budget), parentWork(budget ? budget->used : 0) {
      if (budget)
        budget->used = 0;
    }
    ~LocalScope() {
      if (budget)
        budget->used = parentWork;
    }
    LocalScope(const LocalScope &) = delete;
    LocalScope &operator=(const LocalScope &) = delete;
    LocalScope(LocalScope &&) = delete;
    LocalScope &operator=(LocalScope &&) = delete;

  private:
    SourceCheckBudget *budget;
    uint64_t parentWork;
  };
  /// Temporarily refine a diagnostic/metrics phase inside a larger source
  /// check stage. Nested semantic planners restore their caller's phase on
  /// every exit, including failures.
  class PhaseScope {
  public:
    PhaseScope(SourceCheckBudget &budget, llvm::StringRef phase)
        : PhaseScope(&budget, phase) {}
    PhaseScope(SourceCheckBudget *budget, llvm::StringRef phase)
        : budget(budget), parentPhase(budget ? budget->phase : llvm::StringRef()) {
      if (budget)
        budget->phase = phase;
    }
    ~PhaseScope() {
      if (budget)
        budget->phase = parentPhase;
    }
    PhaseScope(const PhaseScope &) = delete;
    PhaseScope &operator=(const PhaseScope &) = delete;
    PhaseScope(PhaseScope &&) = delete;
    PhaseScope &operator=(PhaseScope &&) = delete;

  private:
    SourceCheckBudget *budget;
    llvm::StringRef parentPhase;
  };
  /// Raw input accounting cannot trust locations or serialize operations.
  /// Restore the previous diagnostic mode on every exit, including nesting.
  class RawDiagnosticScope {
  public:
    explicit RawDiagnosticScope(SourceCheckBudget &budget)
        : budget(budget), previous(budget.rawDiagnostics) {
      budget.rawDiagnostics = true;
    }
    ~RawDiagnosticScope() { budget.rawDiagnostics = previous; }
    RawDiagnosticScope(const RawDiagnosticScope &) = delete;
    RawDiagnosticScope &operator=(const RawDiagnosticScope &) = delete;
    RawDiagnosticScope(RawDiagnosticScope &&) = delete;
    RawDiagnosticScope &operator=(RawDiagnosticScope &&) = delete;

  private:
    SourceCheckBudget &budget;
    bool previous;
  };
  static uint64_t add(uint64_t a, uint64_t b) {
    return b > std::numeric_limits<uint64_t>::max() - a
               ? std::numeric_limits<uint64_t>::max()
               : a + b;
  }
  static uint64_t multiply(uint64_t a, uint64_t b) {
    return a && b > std::numeric_limits<uint64_t>::max() / a
               ? std::numeric_limits<uint64_t>::max()
               : a * b;
  }
  static uint64_t bytes(uint64_t size) { return size / 8 + (size % 8 != 0); }
  // Prepay the height of a red-black tree, including the empty-tree probe.
  static uint64_t orderedLookupCost(uint64_t count) {
    return 1 + 2 * std::bit_width(count);
  }
  static uint64_t orderedInsertCost(uint64_t count, uint64_t valueSize) {
    // Cover the search and rebalancing, plus the node and its bookkeeping.
    return add(4 + 4 * std::bit_width(count),
               bytes(add(valueSize, 4 * sizeof(void *))));
  }
  mlir::LogicalResult admit(uint64_t units, mlir::Operation *site) const {
    // Saturation is always a rejection, even with a maximal explicit limit.
    if (units == std::numeric_limits<uint64_t>::max() || units > limit - used) {
      auto error = rawDiagnostics
                       ? mlir::emitError(mlir::UnknownLoc::get(site->getContext()))
                       : site->emitOpError();
      return error
             << "source-check plan exceeds analysis work budget before "
                "occurrence expansion (requested "
             << units << ", used " << used << ", limit " << limit << ", phase "
             << phase << ")";
    }
    return mlir::success();
  }
  mlir::LogicalResult debit(uint64_t units, mlir::Operation *site) {
    if (mlir::failed(admit(units, site)))
      return mlir::failure();
    used += units;
    peak = std::max(peak, used);
    total = add(total, units);
    if (metrics)
      metrics->details[("work:" + phase).str()] += units;
    return mlir::success();
  }
  mlir::LogicalResult debitPersistent(uint64_t units,
                                      mlir::Operation *site) {
    if (units == std::numeric_limits<uint64_t>::max() ||
        units > limit - persistent) {
      auto error = rawDiagnostics
                       ? mlir::emitError(
                             mlir::UnknownLoc::get(site->getContext()))
                       : site->emitOpError();
      return error
             << "source-check persistent plan storage exceeds analysis work "
                "budget before occurrence expansion (requested "
             << units << ", used " << persistent << ", limit " << limit
             << ", phase " << phase << ")";
    }
    if (mlir::failed(debit(units, site)))
      return mlir::failure();
    persistent += units;
    peak = std::max(peak, persistent);
    return mlir::success();
  }
  void setPhase(llvm::StringRef name) { phase = name; }
  uint64_t usedWorkUnits() const { return std::max(peak, persistent); }
  void setMetrics(acir::metrics::CompilerPhaseMetrics *value,
                  const uint64_t *excludedWork = nullptr) {
    metrics = value;
    excluded = excludedWork;
  }
  ~SourceCheckBudget() {
    if (metrics)
      metrics->work += total - (excluded ? std::min(total, *excluded) : 0);
  }
  static uint64_t bindings(const HardwareBindings &scope) {
    uint64_t size = sizeof(HardwareBindings);
    for (const auto &entry : scope.integers)
      size = add(size, add(sizeof(entry), entry.first().size() + 1));
    for (const auto &entry : scope.types)
      size = add(size, add(sizeof(entry), entry.first().size() + 1));
    return bytes(size);
  }

private:
  uint64_t limit, used = 0, peak = 0, total = 0, persistent = 0;
  llvm::StringRef phase = "initialization";
  bool rawDiagnostics = false;
  acir::metrics::CompilerPhaseMetrics *metrics = nullptr;
  const uint64_t *excluded = nullptr;
};
} // namespace acir::ac::detail
#endif
