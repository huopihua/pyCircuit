#ifndef PYCIRCUIT_FINAL_VERILOG_SOURCE_PARTS_H
#define PYCIRCUIT_FINAL_VERILOG_SOURCE_PARTS_H
#include "FinalSourceMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>
#include <vector>
namespace acir::compiler {
/// Owned source text and publication paths produced by Verilog target lowering.
/// Pure renderers serialize this payload and perform no IR analysis.
struct FinalVerilogSourceGroup {
  FinalSourceOwner sourceOwner;
  llvm::SmallVector<std::string> definitions;
  std::string path, text;
};
struct FinalVerilogSourceParts {
  llvm::SmallVector<FinalVerilogSourceGroup, 0> sourceGroups;
  llvm::SmallVector<std::string> standardSources;
  std::string typePackage, core, runtimeGlue, rootRtlName;
  std::string simulationTop, portMap;
  uint64_t sourceCheckCount = 0;
};
} // namespace acir::compiler
#endif
