#ifndef PYCIRCUIT_ANALYSIS_COMPILER_PHASE_METRICS_H
#define PYCIRCUIT_ANALYSIS_COMPILER_PHASE_METRICS_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace acir::metrics {

/// Non-authoritative measurements from one stable compiler phase. These
/// counters describe work already performed; they never participate in
/// admission, verification, lowering, or emitted semantics.
struct CompilerPhaseMetrics {
  std::string name;
  std::string target;
  std::string status = "incomplete";
  uint64_t work = 0;
  uint64_t slots = 0;
  uint64_t occurrences = 0;
  uint64_t definitions = 0;
  uint64_t calls = 0;
  uint64_t instances = 0;
  uint64_t irWalks = 0;
  uint64_t visitedNodes = 0;
  uint64_t planEntries = 0;
  uint64_t planBytes = 0;
  double wallSeconds = 0.0;
  llvm::StringMap<uint64_t> details;
};

struct CompilerTimingMetrics {
  double compileSeconds = 0.0;
  double linkSeconds = 0.0;
  double emitCppSeconds = 0.0;
  double emitVerilogSeconds = 0.0;
};

struct CompilerMetricsReport {
  std::string status = "incomplete";
  std::string inputPath;
  std::string inputSha256;
  std::vector<CompilerPhaseMetrics> phases;
  CompilerTimingMetrics timing;

  CompilerPhaseMetrics &phase(llvm::StringRef name,
                              llvm::StringRef target = {});
};

enum class CompilerTimingKind {
  Compile,
  Link,
  EmitCpp,
  EmitVerilog,
};

/// Close one tool-level metrics report exactly once. An unfinished guard writes
/// a failed report when an output path was requested, so early returns retain
/// diagnostic evidence without changing compiler artifacts.
class CompilerMetricsGuard {
public:
  CompilerMetricsGuard(llvm::StringRef outputPath,
                       CompilerMetricsReport &report, CompilerTimingKind timing,
                       CompilerPhaseMetrics *phase = nullptr);
  ~CompilerMetricsGuard();

  CompilerMetricsGuard(const CompilerMetricsGuard &) = delete;
  CompilerMetricsGuard &operator=(const CompilerMetricsGuard &) = delete;

  bool finish(bool succeeded);

private:
  std::string outputPath;
  CompilerMetricsReport &report;
  CompilerTimingKind timing;
  CompilerPhaseMetrics *phase;
  std::chrono::steady_clock::time_point start;
  bool finished = false;
  bool writeSucceeded = true;
};

/// Write one closed v1 report to a fresh ordinary file. The report channel is
/// diagnostic evidence and is deliberately separate from compiler artifacts.
bool writeCompilerMetrics(llvm::StringRef path,
                          const CompilerMetricsReport &report);

} // namespace acir::metrics

#endif // PYCIRCUIT_ANALYSIS_COMPILER_PHASE_METRICS_H
