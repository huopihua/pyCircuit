#include "BackendRenderPlan.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <set>

using namespace mlir;

namespace acir::compiler {
namespace {

llvm::json::Object sourceOwner(const FinalSourceOwner &owner) {
  return llvm::json::Object{{"package", owner.package}, {"path", owner.path}};
}

FailureOr<unsigned> membershipIndex(ArrayRef<FinalSourceMembership> memberships,
                                    const FinalSourceOwner &owner,
                                    ac::detail::EmitError emitError) {
  auto found = llvm::find_if(memberships, [&](const auto &membership) {
    return membership.sourceOwner == owner;
  });
  if (found == memberships.end())
    return emitError() << "native source group has no final SourceOwner";
  return static_cast<unsigned>(found - memberships.begin());
}

LogicalResult verifyMetadata(const HardwareRenderMetadata &metadata,
                             ac::detail::EmitError emitError) {
  if (metadata.implementationDefinition.empty() ||
      metadata.logicalDefinition.empty() || metadata.entrySource.empty())
    return emitError() << "render plan has an invalid entry";
  if (metadata.sourceMaps.empty())
    return emitError() << "render plan has no source maps";
  std::set<std::pair<std::string, std::string>> owners;
  std::set<std::string> paths;
  for (const FinalSourceMap &map : metadata.sourceMaps) {
    if (map.sourceOwner.empty() ||
        !owners.insert({map.sourceOwner.package, map.sourceOwner.path}).second ||
        map.path.empty() || !paths.insert(map.path).second)
      return emitError() << "render plan has duplicate source-map identity";
    auto parsed = llvm::json::parse(map.text);
    auto parsedObject = parsed ? parsed->getAsObject() : nullptr;
    if (!llvm::is_sorted(map.generatedFiles) ||
        llvm::adjacent_find(map.generatedFiles) != map.generatedFiles.end())
      return emitError()
             << "render plan source-map membership is not sorted and unique";
    auto object = parsedObject;
    auto source = object ? object->getObject("source") : nullptr;
    auto files = object ? object->getArray("generated_files") : nullptr;
    auto origins = object ? object->getArray("origins") : nullptr;
    auto kind = object ? object->getString("kind") : std::nullopt;
    auto package = source ? source->getString("package") : std::nullopt;
    auto path = source ? source->getString("path") : std::nullopt;
    if (!object || object->size() != 4 || !kind ||
        *kind != "pycircuit-source-map" || !source || source->size() != 2 ||
        !files || !origins || !package || !path ||
        *package != map.sourceOwner.package || *path != map.sourceOwner.path ||
        files->size() != map.generatedFiles.size()) {
      if (!parsed)
        llvm::consumeError(parsed.takeError());
      return emitError()
             << "render plan source-map payload differs from frozen metadata";
    }
    for (auto [index, file] : llvm::enumerate(*files)) {
      auto text = file.getAsString();
      if (!text || *text != map.generatedFiles[index])
        return emitError()
               << "render plan source-map payload differs from membership";
    }
    for (const llvm::json::Value &origin : *origins) {
      auto row = origin.getAsObject();
      auto operation = row ? row->getString("operation") : std::nullopt;
      auto occurrence = row ? row->getString("origin") : std::nullopt;
      auto location = row ? row->getString("location") : std::nullopt;
      if (!row || row->size() != 3 || !operation || operation->empty() ||
          !occurrence || occurrence->empty() || !location ||
          location->empty())
        return emitError()
               << "render plan source-map has an invalid frozen origin";
    }
  }
  return success();
}

LogicalResult verifyPortMap(StringRef text, ac::detail::EmitError emitError) {
  auto parsed = llvm::json::parse(text);
  if (!parsed || !parsed->getAsObject()) {
    if (!parsed)
      llvm::consumeError(parsed.takeError());
    return emitError() << "render plan has an invalid root port map";
  }
  return success();
}

LogicalResult verifyMembership(const HardwareRenderMetadata &metadata,
                               ArrayRef<FinalSourceMembership> memberships,
                               ac::detail::EmitError emitError) {
  if (metadata.sourceMaps.size() != memberships.size())
    return emitError() << "render plan source-map inventory is incomplete";
  for (const FinalSourceMap &map : metadata.sourceMaps) {
    auto index = membershipIndex(memberships, map.sourceOwner, emitError);
    if (failed(index))
      return failure();
    SmallVector<std::string> files = memberships[*index].generatedFiles;
    llvm::sort(files);
    if (files != map.generatedFiles)
      return emitError()
             << "render plan source-map membership differs from target files";
  }
  return success();
}

LogicalResult addPublicationPath(StringRef path, std::set<std::string> &paths,
                                 ac::detail::EmitError emitError) {
  if (path.empty() || path.front() == '/' || path.contains('\\') ||
      path.contains('\0') || path.contains(':'))
    return emitError() << "render plan contains an unsafe publication path";
  SmallVector<StringRef> components;
  path.split(components, '/', -1, true);
  if (llvm::any_of(components, [](StringRef component) {
        return component.empty() || component == "." || component == "..";
      }))
    return emitError() << "render plan contains an unsafe publication path";
  if (!paths.insert(path.str()).second)
    return emitError() << "render plan publication paths collide";
  return success();
}

llvm::json::Object commonJson(const HardwareRenderMetadata &metadata) {
  llvm::json::Array arguments;
  for (const auto &argument : metadata.logicalArguments)
    arguments.emplace_back(
        llvm::json::Object{{"kind", argument.kind}, {"value", argument.value}});
  llvm::json::Object result{
      {"entry",
       llvm::json::Object{
           {"definition", "@\"" + metadata.logicalDefinition + "\""},
           {"arguments", std::move(arguments)}}},
      {"entry_source", sourceOwner(metadata.entrySource)},
      {"reset_semantics",
       metadata.resetSemantics == ac::HardwareResetSemantics::IfOne ? "if_one"
                                                                    : "strict"},
      {"needs_four_state", metadata.needsFourState}};
  if (metadata.execution) {
    result["execution_model"] = llvm::json::Object{
        {"strategy", metadata.execution->strategy},
        {"protocol", metadata.execution->protocol},
        {"publication_field", metadata.execution->publicationField}};
    if (metadata.execution->stateOutput)
      result["execution_model"].getAsObject()->insert(
          {"state_output", *metadata.execution->stateOutput});
  }
  llvm::json::Array samples;
  for (const std::string &row : metadata.systemSamples)
    samples.emplace_back(row);
  result["system_samples"] = std::move(samples);
  return result;
}

void addSourceMaps(llvm::json::Object &result,
                   const HardwareRenderMetadata &metadata) {
  llvm::json::Array maps;
  for (const FinalSourceMap &map : metadata.sourceMaps)
    maps.emplace_back(
        llvm::json::Object{{"source", sourceOwner(map.sourceOwner)},
                           {"path", map.path},
                           {"text", map.text}});
  result["source_maps"] = std::move(maps);
}

void printJson(llvm::json::Object result, llvm::raw_ostream &out) {
  out << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(result)));
}

} // namespace

LogicalResult verifyCppRenderPlan(const CppRenderPlan &plan,
                                  ac::detail::EmitError emitError) {
  if (failed(verifyMetadata(plan.metadata, emitError)) ||
      plan.parts.supportHeader.empty() || plan.parts.systemHeader.empty() ||
      plan.parts.rootCppName.empty() || plan.parts.rootHeaderPath.empty() ||
      failed(verifyPortMap(plan.parts.portMap, emitError)))
    return emitError() << "C++ render plan is incomplete";
  std::set<std::string> paths;
  for (StringRef fixed :
       {StringRef("pycircuit_support.hpp"), StringRef("pycircuit_system.hpp")})
    if (failed(addPublicationPath(fixed, paths, emitError)))
      return failure();
  if (!plan.parts.simulationMain.empty() &&
      (failed(addPublicationPath("simulation_main.cpp", paths, emitError)) ||
       failed(addPublicationPath("simulation_config.json", paths, emitError))))
    return failure();
  SmallVector<FinalSourceMembership> memberships;
  llvm::StringMap<unsigned> indices;
  auto ownerKey = [](const FinalSourceOwner &owner) {
    return owner.package + "\n" + owner.path;
  };
  for (const FinalSourceMap &map : plan.metadata.sourceMaps) {
    if (!indices.try_emplace(ownerKey(map.sourceOwner), memberships.size())
             .second)
      return emitError() << "C++ render plan repeats a SourceOwner";
    memberships.push_back({map.sourceOwner, {}});
  }
  for (const FinalCppSourceGroup &group : plan.parts.sourceGroups) {
    auto found = indices.find(ownerKey(group.sourceOwner));
    if (group.sourceOwner.empty() || found == indices.end() ||
        group.header.empty() ||
        group.headerPath.empty() ||
        failed(addPublicationPath(group.headerPath, paths, emitError)))
      return emitError() << "C++ render plan has an invalid source group";
    memberships[found->second].generatedFiles.push_back(group.headerPath);
    if (group.sourcePath.empty()) {
      if (!group.source.empty())
        return emitError()
               << "C++ header-only source group has implementation text";
    } else {
      if (group.source.empty() ||
          failed(addPublicationPath(group.sourcePath, paths, emitError)))
        return emitError() << "C++ render plan has an invalid implementation";
      memberships[found->second].generatedFiles.push_back(group.sourcePath);
    }
  }
  for (const FinalSourceMap &map : plan.metadata.sourceMaps)
    if (failed(addPublicationPath(map.path, paths, emitError)))
      return failure();
  return verifyMembership(plan.metadata, memberships, emitError);
}

FailureOr<VerifiedCppRenderPlan>
verifyAndFreezeCppRenderPlan(CppRenderPlan plan,
                             ac::detail::EmitError emitError) {
  if (failed(verifyCppRenderPlan(plan, emitError)))
    return failure();
  return VerifiedCppRenderPlan(std::move(plan));
}

LogicalResult verifyVerilogRenderPlan(const VerilogRenderPlan &plan,
                                      ac::detail::EmitError emitError) {
  if (failed(verifyMetadata(plan.metadata, emitError)) ||
      plan.parts.core.empty() || plan.parts.rootRtlName.empty() ||
      failed(verifyPortMap(plan.parts.portMap, emitError)))
    return emitError() << "Verilog render plan is incomplete";
  std::set<std::string> paths;
  if (failed(addPublicationPath("design_top.sv", paths, emitError)) ||
      (!plan.parts.simulationTop.empty() &&
       failed(addPublicationPath("simulation_top.sv", paths, emitError))))
    return failure();
  SmallVector<FinalSourceMembership> memberships;
  llvm::StringMap<unsigned> indices;
  auto ownerKey = [](const FinalSourceOwner &owner) {
    return owner.package + "\n" + owner.path;
  };
  for (const FinalSourceMap &map : plan.metadata.sourceMaps) {
    if (!indices.try_emplace(ownerKey(map.sourceOwner), memberships.size())
             .second)
      return emitError() << "Verilog render plan repeats a SourceOwner";
    memberships.push_back({map.sourceOwner, {}});
  }
  llvm::StringSet<> standardSources;
  for (StringRef source : plan.parts.standardSources)
    if (!source.starts_with("include/verilog/") ||
        !standardSources.insert(source).second)
      return emitError()
             << "Verilog render plan has an invalid standard source";
  for (const FinalVerilogSourceGroup &group : plan.parts.sourceGroups) {
    auto found = indices.find(ownerKey(group.sourceOwner));
    if (group.sourceOwner.empty() || found == indices.end() ||
        group.path.empty() ||
        group.text.empty() ||
        failed(addPublicationPath(group.path, paths, emitError)))
      return emitError() << "Verilog render plan has an invalid source group";
    memberships[found->second].generatedFiles.push_back(group.path);
  }
  for (const FinalSourceMap &map : plan.metadata.sourceMaps)
    if (failed(addPublicationPath(map.path, paths, emitError)))
      return failure();
  return verifyMembership(plan.metadata, memberships, emitError);
}

FailureOr<VerifiedVerilogRenderPlan>
verifyAndFreezeVerilogRenderPlan(VerilogRenderPlan plan,
                                 ac::detail::EmitError emitError) {
  if (failed(verifyVerilogRenderPlan(plan, emitError)))
    return failure();
  return VerifiedVerilogRenderPlan(std::move(plan));
}

void renderCppSource(const VerifiedCppRenderPlan &plan,
                     llvm::raw_ostream &out) {
  const FinalCppSourceParts &parts = plan->parts;
  out << parts.supportHeader;
  for (const FinalCppSourceGroup &group : parts.sourceGroups)
    out << "\n" << group.header;
  out << "\n" << parts.systemHeader;
}

void renderVerilogSource(const VerifiedVerilogRenderPlan &plan,
                         llvm::raw_ostream &out) {
  const FinalVerilogSourceParts &parts = plan->parts;
  out << parts.core;
  for (const FinalVerilogSourceGroup &group : parts.sourceGroups)
    out << group.text;
  out << parts.runtimeGlue;
  out << parts.simulationTop;
}

void renderCppJson(const VerifiedCppRenderPlan &plan, llvm::raw_ostream &out) {
  const CppRenderPlan &render = plan.get();
  llvm::json::Object result = commonJson(render.metadata);
  llvm::json::Array groups;
  for (const FinalCppSourceGroup &group : render.parts.sourceGroups)
    groups.emplace_back(llvm::json::Object{
        {"source", sourceOwner(group.sourceOwner)},
        {"header_path", group.headerPath},
        {"source_path", group.sourcePath.empty()
                            ? llvm::json::Value(nullptr)
                            : llvm::json::Value(group.sourcePath)},
        {"header", group.header},
        {"implementation", group.sourcePath.empty()
                               ? llvm::json::Value(nullptr)
                               : llvm::json::Value(group.source)}});
  result["support_header"] = render.parts.supportHeader;
  result["system_header"] = render.parts.systemHeader;
  result["simulation_main"] = render.parts.simulationMain;
  result["simulation_config"] = render.parts.simulationConfig;
  result["source_check_count"] = render.parts.sourceCheckCount;
  result["root_cpp_name"] = render.parts.rootCppName;
  auto ports = llvm::json::parse(render.parts.portMap);
  if (ports)
    result["port_availability"] = std::move(*ports);
  else
    llvm::consumeError(ports.takeError());
  result["source_groups"] = std::move(groups);
  addSourceMaps(result, render.metadata);
  printJson(std::move(result), out);
}

void renderVerilogJson(const VerifiedVerilogRenderPlan &plan,
                       llvm::raw_ostream &out) {
  const VerilogRenderPlan &render = plan.get();
  llvm::json::Object result = commonJson(render.metadata);
  llvm::json::Array groups, standard;
  for (const FinalVerilogSourceGroup &group : render.parts.sourceGroups)
    groups.emplace_back(
        llvm::json::Object{{"source", sourceOwner(group.sourceOwner)},
                           {"path", group.path},
                           {"text", group.text}});
  for (const std::string &source : render.parts.standardSources)
    standard.emplace_back(source);
  result["rtl_source_groups"] = std::move(groups);
  result["rtl_standard_sources"] = std::move(standard);
  result["rtl_core"] = render.parts.core;
  result["rtl_mode"] = render.physical ? "physical" : "managed";
  result["simulation_top"] = render.parts.simulationTop;
  result["source_check_count"] = render.parts.sourceCheckCount;
  result["root_rtl_name"] = render.parts.rootRtlName;
  auto ports = llvm::json::parse(render.parts.portMap);
  if (ports)
    result["port_availability"] = std::move(*ports);
  else
    llvm::consumeError(ports.takeError());
  addSourceMaps(result, render.metadata);
  printJson(std::move(result), out);
}

} // namespace acir::compiler
