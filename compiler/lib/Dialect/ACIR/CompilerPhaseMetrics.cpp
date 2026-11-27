#include "pycircuit/Analysis/CompilerPhaseMetrics.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace acir::metrics {

CompilerPhaseMetrics &CompilerMetricsReport::phase(llvm::StringRef name,
                                                   llvm::StringRef target) {
  for (auto &entry : phases)
    if (entry.name == name && entry.target == target)
      return entry;
  phases.push_back({});
  phases.back().name = name.str();
  phases.back().target = target.str();
  return phases.back();
}

CompilerMetricsGuard::CompilerMetricsGuard(llvm::StringRef outputPath,
                                           CompilerMetricsReport &report,
                                           CompilerTimingKind timing,
                                           CompilerPhaseMetrics *phase)
    : outputPath(outputPath.str()), report(report), timing(timing),
      phase(phase), start(std::chrono::steady_clock::now()) {}

CompilerMetricsGuard::~CompilerMetricsGuard() {
  if (!finished)
    (void)finish(false);
}

bool CompilerMetricsGuard::finish(bool succeeded) {
  if (finished)
    return writeSucceeded;
  finished = true;
  report.status = succeeded ? "complete" : "failed";
  double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  if (phase) {
    phase->status = report.status;
    phase->wallSeconds = elapsed;
  }
  switch (timing) {
  case CompilerTimingKind::Compile:
    report.timing.compileSeconds = elapsed;
    break;
  case CompilerTimingKind::Link:
    report.timing.linkSeconds = elapsed;
    break;
  case CompilerTimingKind::EmitCpp:
    report.timing.emitCppSeconds = elapsed;
    break;
  case CompilerTimingKind::EmitVerilog:
    report.timing.emitVerilogSeconds = elapsed;
    break;
  }
  writeSucceeded =
      outputPath.empty() || writeCompilerMetrics(outputPath, report);
  return writeSucceeded;
}

bool writeCompilerMetrics(llvm::StringRef path,
                          const CompilerMetricsReport &report) {
  llvm::json::Array phases;
  for (const auto &phase : report.phases) {
    llvm::json::Object details;
    for (const auto &entry : phase.details)
      details[entry.getKey()] = entry.getValue();
    phases.emplace_back(llvm::json::Object{
        {"name", phase.name},
        {"target", phase.target.empty() ? llvm::json::Value(nullptr)
                                        : llvm::json::Value(phase.target)},
        {"status", phase.status},
        {"work", phase.work},
        {"slots", phase.slots},
        {"occurrences", phase.occurrences},
        {"definitions", phase.definitions},
        {"calls", phase.calls},
        {"instances", phase.instances},
        {"ir_walks", phase.irWalks},
        {"visited_nodes", phase.visitedNodes},
        {"plan_entries", phase.planEntries},
        {"plan_bytes", phase.planBytes},
        {"wall_seconds", phase.wallSeconds},
        {"details", std::move(details)},
    });
  }
  llvm::json::Object input{{"path", report.inputPath},
                           {"sha256", report.inputSha256}};
  llvm::json::Object timing{
      {"compile_seconds", report.timing.compileSeconds},
      {"link_seconds", report.timing.linkSeconds},
      {"emit_cpp_seconds", report.timing.emitCppSeconds},
      {"emit_verilog_seconds", report.timing.emitVerilogSeconds},
  };
  llvm::json::Object root{
      {"schema", "pycircuit-compiler-phase-metrics-v1"},
      {"status", report.status},
      {"input", std::move(input)},
      {"phases", std::move(phases)},
      {"timing", std::move(timing)},
  };
  int descriptor = -1;
  std::error_code error = llvm::sys::fs::openFileForWrite(
      path, descriptor, llvm::sys::fs::CD_CreateNew, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << path << ": " << error.message() << '\n';
    return false;
  }
  llvm::raw_fd_ostream output(descriptor, true);
  llvm::sys::fs::file_status status;
  if (std::error_code statusError = llvm::sys::fs::status(descriptor, status)) {
    llvm::errs() << path << ": " << statusError.message() << '\n';
    return false;
  }
  if (!llvm::sys::fs::is_regular_file(status)) {
    llvm::errs() << path << ": metrics output must be an ordinary file\n";
    return false;
  }
  output << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
  output.flush();
  if (output.has_error()) {
    llvm::errs() << path << ": " << output.error().message() << '\n';
    output.clear_error();
    return false;
  }
  return true;
}

} // namespace acir::metrics
