#ifndef PYCIRCUIT_FINAL_CPP_SOURCE_PARTS_H
#define PYCIRCUIT_FINAL_CPP_SOURCE_PARTS_H
#include "FinalSourceMap.h"
#include <cstdint>
#include <string>
#include <vector>
namespace acir::compiler {
/// Owned source text and publication paths produced by C++ target lowering.
/// Pure renderers serialize this payload and perform no IR analysis.
struct FinalCppSourceGroup {
  FinalSourceOwner sourceOwner;
  std::string headerPath, sourcePath, header, source;
};
struct FinalCppSourceParts {
  std::string supportHeader, nominalDeclarations, systemHeader, rootCppName,
      rootHeaderPath;
  std::vector<FinalCppSourceGroup> sourceGroups;
  std::string simulationMain, simulationConfig, portMap;
  uint64_t sourceCheckCount = 0;
};
} // namespace acir::compiler
#endif
