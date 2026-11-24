#include "Compiler/HardwareEmissionPlan.h"
#include "Compiler/BackendRenderPlan.h"

#include "mlir/Parser/Parser.h"
#include "pycircuit/Dialect/ACIR/ACIRDialect.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

#include <type_traits>

namespace {
namespace ac = acir::ac;
namespace compiler = acir::compiler;
using namespace mlir;

class HardwareEmissionPlanTest : public ::testing::Test {
protected:
  MLIRContext context;

  void SetUp() override { context.loadDialect<ac::ACIRDialect>(); }

  OwningOpRef<ModuleOp> parse(StringRef relative) {
    SmallString<256> path(PYCIRCUIT_TEST_REPO_ROOT);
    llvm::sys::path::append(path, relative);
    return parseSourceFile<ModuleOp>(path, &context);
  }
};

static_assert(
    !std::is_copy_constructible_v<compiler::VerifiedHardwareEmissionPlan>);
static_assert(
    std::is_move_constructible_v<compiler::VerifiedHardwareEmissionPlan>);
static_assert(!std::is_copy_constructible_v<compiler::VerifiedCppRenderPlan>);
static_assert(std::is_move_constructible_v<compiler::VerifiedCppRenderPlan>);
static_assert(
    !std::is_copy_constructible_v<compiler::VerifiedVerilogRenderPlan>);
static_assert(
    std::is_move_constructible_v<compiler::VerifiedVerilogRenderPlan>);
static_assert(std::is_same_v<
              decltype(compiler::FinalSourceMap::sourceOwner),
              compiler::FinalSourceOwner>);
static_assert(std::is_same_v<
              decltype(compiler::FinalCppSourceGroup::sourceOwner),
              compiler::FinalSourceOwner>);
static_assert(std::is_same_v<
              decltype(compiler::FinalVerilogSourceGroup::sourceOwner),
              compiler::FinalSourceOwner>);
static_assert(std::is_same_v<
              decltype(compiler::FinalVerilogSourceGroup::definitions),
              llvm::SmallVector<std::string>>);

TEST_F(HardwareEmissionPlanTest, FreezesFinalClosureChecksAndMetadata) {
  auto package =
      parse("tests/compiler/lit/CodeGen/Inputs/sampled-admission.mlir");
  ASSERT_TRUE(package);

  auto plan = compiler::prepareHardwareEmissionPlan(*package);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->get().entry.implementationDefinition, "Top");
  EXPECT_EQ(plan->get().entry.logicalDefinition, "Top");
  ASSERT_EQ(plan->get().modules.size(), 4u);
  EXPECT_FALSE(plan->get().definitions.empty());
  for (const auto &definition : plan->get().definitions) {
    EXPECT_NE(definition.diagnosticAnchor, nullptr);
    EXPECT_FALSE(definition.names.cppQualified.empty());
    EXPECT_FALSE(definition.names.rtl.empty());
    EXPECT_FALSE(definition.names.header.empty());
    EXPECT_FALSE(definition.names.verilog.empty());
  }
  EXPECT_EQ(plan->get().modules[plan->get().rootModule].key.definition, "Top");
  EXPECT_TRUE(plan->get().needsFourState);
  EXPECT_FALSE(plan->get().checks.empty());
  ASSERT_EQ(plan->get().sourceUnits.size(), 1u);
  EXPECT_EQ(plan->get().sourceUnits.front().moduleIndices.size(), 4u);
  EXPECT_EQ(plan->get().sourceUnits.front().definitionIndices.size(),
            plan->get().definitions.size());
  EXPECT_FALSE(plan->get().sourceUnits.front().cppHeaderPath.empty());
  EXPECT_FALSE(plan->get().sourceUnits.front().cppSourcePath.empty());
  EXPECT_FALSE(plan->get().sourceUnits.front().verilogPath.empty());
  EXPECT_FALSE(plan->get().primitives.empty());
  EXPECT_FALSE(plan->get().standardVerilogSources.empty());
  ASSERT_EQ(plan->get().sourceMaps.size(), 1u);
  EXPECT_EQ(plan->get().sourceMaps.front().sourceOwner,
            plan->get().sourceUnits.front().sourceOwner);
  EXPECT_FALSE(plan->get().sourceMaps.front().path.empty());
  EXPECT_FALSE(plan->get().sourceMaps.front().origins.empty());

  for (const auto &module : plan->get().modules) {
    ASSERT_FALSE(module.values.empty());
    for (auto [index, value] : llvm::enumerate(module.values)) {
      EXPECT_EQ(value.id, index);
      EXPECT_GT(value.type.packedWidth, 0u);
      EXPECT_FALSE(value.type.canonicalType.empty());
    }
    for (auto [stepIndex, step] : llvm::enumerate(module.schedule)) {
      EXPECT_EQ(step.id, stepIndex);
      EXPECT_NE(step.diagnosticAnchor, nullptr);
      EXPECT_FALSE(step.opcode.empty());
      for (auto value : step.inputs)
        EXPECT_LT(value, module.values.size());
      for (auto value : step.outputs)
        EXPECT_LT(value, module.values.size());
    }
  }
}

TEST_F(HardwareEmissionPlanTest, RejectsMutatedFrozenSourceMapProvenance) {
  auto package =
      parse("tests/compiler/lit/CodeGen/Inputs/sampled-admission.mlir");
  ASSERT_TRUE(package);

  auto prepared = compiler::prepareHardwareEmissionPlan(*package);
  ASSERT_TRUE(succeeded(prepared));
  compiler::HardwareEmissionPlan mutated = prepared->get();
  ASSERT_FALSE(mutated.sourceMaps.empty());
  mutated.sourceMaps.front().path = "../escaped.source-map.json";
  EXPECT_TRUE(failed(compiler::verifyHardwareEmissionPlan(mutated)));

  mutated = prepared->get();
  ASSERT_FALSE(mutated.sourceMaps.front().origins.empty());
  mutated.sourceMaps.front().origins.front().location.clear();
  EXPECT_TRUE(failed(compiler::verifyHardwareEmissionPlan(mutated)));
}

TEST_F(HardwareEmissionPlanTest, RejectsMutatedAvailabilityInventory) {
  auto package =
      parse("tests/compiler/lit/CodeGen/Inputs/sampled-admission.mlir");
  ASSERT_TRUE(package);

  auto prepared = compiler::prepareHardwareEmissionPlan(*package);
  ASSERT_TRUE(succeeded(prepared));
  compiler::HardwareEmissionPlan mutated = prepared->get();
  auto &root = mutated.modules[mutated.rootModule];
  ASSERT_FALSE(root.outputs.empty());
  root.outputs.front().availabilityOutput = 0;
  EXPECT_TRUE(failed(compiler::verifyHardwareEmissionPlan(mutated)));
}

TEST_F(HardwareEmissionPlanTest, FreezesAndPureRendersBothTargets) {
  auto package =
      parse("tests/compiler/lit/CodeGen/Inputs/sampled-admission.mlir");
  ASSERT_TRUE(package);
  auto hardware = compiler::prepareHardwareEmissionPlan(*package);
  ASSERT_TRUE(succeeded(hardware));

  auto cpp = compiler::prepareCppRenderPlan(*hardware);
  auto verilog = compiler::prepareVerilogRenderPlan(*hardware);
  ASSERT_TRUE(succeeded(cpp));
  ASSERT_TRUE(succeeded(verilog));
  EXPECT_EQ(cpp->get().metadata.sourceMaps.size(),
            hardware->get().sourceMaps.size());
  EXPECT_EQ(verilog->get().metadata.sourceMaps.size(),
            hardware->get().sourceMaps.size());

  std::string cppJson, cppSource, rtlJson, rtlSource;
  llvm::raw_string_ostream cppJsonOut(cppJson), cppSourceOut(cppSource),
      rtlJsonOut(rtlJson), rtlSourceOut(rtlSource);
  compiler::renderCppJson(*cpp, cppJsonOut);
  compiler::renderCppSource(*cpp, cppSourceOut);
  compiler::renderVerilogJson(*verilog, rtlJsonOut);
  compiler::renderVerilogSource(*verilog, rtlSourceOut);
  cppJsonOut.flush();
  cppSourceOut.flush();
  rtlJsonOut.flush();
  rtlSourceOut.flush();
  EXPECT_TRUE(llvm::json::parse(cppJson));
  EXPECT_TRUE(llvm::json::parse(rtlJson));
  EXPECT_FALSE(cppSource.empty());
  EXPECT_FALSE(rtlSource.empty());
}

TEST_F(HardwareEmissionPlanTest,
       OwnsFinalSnapshotAcrossSourceMutationAndDestruction) {
  auto package =
      parse("tests/compiler/lit/CodeGen/Inputs/sampled-admission.mlir");
  ASSERT_TRUE(package);
  ModuleOp input = *package;
  auto hardware = compiler::prepareHardwareEmissionPlan(*package);
  ASSERT_TRUE(succeeded(hardware));
  ASSERT_NE(hardware->get().package, input);

  (*package)->setAttr("ac.stage", StringAttr::get(&context, "source"));
  package = OwningOpRef<ModuleOp>();

  auto cpp = compiler::prepareCppRenderPlan(*hardware);
  auto verilog = compiler::prepareVerilogRenderPlan(*hardware);
  ASSERT_TRUE(succeeded(cpp));
  ASSERT_TRUE(succeeded(verilog));
  EXPECT_FALSE(cpp->get().parts.supportHeader.empty());
  EXPECT_FALSE(verilog->get().parts.core.empty());
}

TEST(HardwareEmissionPlanLifetimeTest, OwnsItsMlirContext) {
  std::optional<compiler::VerifiedHardwareEmissionPlan> hardware;
  {
    MLIRContext sourceContext;
    sourceContext.loadDialect<ac::ACIRDialect>();
    SmallString<256> path(PYCIRCUIT_TEST_REPO_ROOT);
    llvm::sys::path::append(
        path, "tests/compiler/lit/CodeGen/Inputs/sampled-admission.mlir");
    auto package = parseSourceFile<ModuleOp>(path, &sourceContext);
    ASSERT_TRUE(package);
    auto prepared = compiler::prepareHardwareEmissionPlan(*package);
    ASSERT_TRUE(succeeded(prepared));
    hardware.emplace(std::move(*prepared));
  }

  auto cpp = compiler::prepareCppRenderPlan(*hardware);
  auto verilog = compiler::prepareVerilogRenderPlan(*hardware);
  ASSERT_TRUE(succeeded(cpp));
  ASSERT_TRUE(succeeded(verilog));
  EXPECT_FALSE(cpp->get().parts.supportHeader.empty());
  EXPECT_FALSE(verilog->get().parts.core.empty());
}

TEST(HardwareEmissionPlanLifetimeTest,
     RenderPlansOutliveHardwarePlanAndPrivateContext) {
  std::optional<compiler::VerifiedCppRenderPlan> cpp;
  std::optional<compiler::VerifiedVerilogRenderPlan> verilog;
  {
    MLIRContext sourceContext;
    sourceContext.loadDialect<ac::ACIRDialect>();
    SmallString<256> path(PYCIRCUIT_TEST_REPO_ROOT);
    llvm::sys::path::append(
        path, "tests/compiler/lit/CodeGen/Inputs/sampled-admission.mlir");
    auto package = parseSourceFile<ModuleOp>(path, &sourceContext);
    ASSERT_TRUE(package);
    auto hardware = compiler::prepareHardwareEmissionPlan(*package);
    ASSERT_TRUE(succeeded(hardware));
    auto preparedCpp = compiler::prepareCppRenderPlan(*hardware);
    auto preparedVerilog = compiler::prepareVerilogRenderPlan(*hardware);
    ASSERT_TRUE(succeeded(preparedCpp));
    ASSERT_TRUE(succeeded(preparedVerilog));
    cpp.emplace(std::move(*preparedCpp));
    verilog.emplace(std::move(*preparedVerilog));
  }

  std::string cppJson, cppSource, rtlJson, rtlSource;
  llvm::raw_string_ostream cppJsonOut(cppJson), cppSourceOut(cppSource),
      rtlJsonOut(rtlJson), rtlSourceOut(rtlSource);
  compiler::renderCppJson(*cpp, cppJsonOut);
  compiler::renderCppSource(*cpp, cppSourceOut);
  compiler::renderVerilogJson(*verilog, rtlJsonOut);
  compiler::renderVerilogSource(*verilog, rtlSourceOut);
  cppJsonOut.flush();
  cppSourceOut.flush();
  rtlJsonOut.flush();
  rtlSourceOut.flush();
  EXPECT_TRUE(llvm::json::parse(cppJson));
  EXPECT_TRUE(llvm::json::parse(rtlJson));
  EXPECT_FALSE(cppSource.empty());
  EXPECT_FALSE(rtlSource.empty());
}

TEST_F(HardwareEmissionPlanTest, RejectsMutatedTargetPublicationInventory) {
  auto package =
      parse("tests/compiler/lit/CodeGen/Inputs/sampled-admission.mlir");
  ASSERT_TRUE(package);
  auto hardware = compiler::prepareHardwareEmissionPlan(*package);
  ASSERT_TRUE(succeeded(hardware));
  auto cpp = compiler::prepareCppRenderPlan(*hardware);
  auto verilog = compiler::prepareVerilogRenderPlan(*hardware);
  ASSERT_TRUE(succeeded(cpp));
  ASSERT_TRUE(succeeded(verilog));

  compiler::CppRenderPlan mutatedCpp = cpp->get();
  ASSERT_FALSE(mutatedCpp.parts.sourceGroups.empty());
  mutatedCpp.parts.sourceGroups.front().headerPath = "pycircuit_system.hpp";
  EXPECT_TRUE(failed(compiler::verifyCppRenderPlan(
      mutatedCpp, [&] { return package->emitOpError(); })));

  compiler::VerilogRenderPlan mutatedVerilog = verilog->get();
  ASSERT_FALSE(mutatedVerilog.metadata.sourceMaps.empty());
  mutatedVerilog.metadata.sourceMaps.front().generatedFiles.clear();
  EXPECT_TRUE(failed(compiler::verifyVerilogRenderPlan(
      mutatedVerilog, [&] { return package->emitOpError(); })));
}

TEST_F(HardwareEmissionPlanTest, RetainsDistinctGenericSpecializations) {
  auto package =
      parse("tests/compiler/lit/CodeGen/Inputs/generic-forward.mlir");
  ASSERT_TRUE(package);
  Builder builder(&context);
  (*package)->setAttr("ac.stage", builder.getStringAttr("final"));

  auto plan = compiler::prepareHardwareEmissionPlan(*package);
  ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->get().modules.size(), 3u);
  const auto &root = plan->get().modules[plan->get().rootModule];
  ASSERT_FALSE(plan->get().reachableDefinitions.empty());
  for (const auto &unit : plan->get().sourceUnits)
    EXPECT_EQ(unit.emissionOrder.size(), unit.definitionIndices.size());
  ASSERT_EQ(root.children.size(), 2u);
  ASSERT_TRUE(root.children[0].moduleIndex);
  ASSERT_TRUE(root.children[1].moduleIndex);
  EXPECT_NE(*root.children[0].moduleIndex, *root.children[1].moduleIndex);
  const auto &scalar = plan->get().modules[*root.children[0].moduleIndex];
  const auto &aggregate = plan->get().modules[*root.children[1].moduleIndex];
  ASSERT_EQ(scalar.key.types.size(), 1u);
  ASSERT_EQ(aggregate.key.types.size(), 1u);
  EXPECT_NE(scalar.key.types.front().value, aggregate.key.types.front().value);

  auto generic = llvm::find_if(plan->get().definitions, [](const auto &item) {
    return item.symbol == "identity";
  });
  ASSERT_NE(generic, plan->get().definitions.end());
  ASSERT_FALSE(generic->templateSchedule.empty());
  for (Operation *operation : generic->templateSchedule)
    EXPECT_EQ(operation->getParentOfType<ac::ModuleOp>(),
              generic->diagnosticAnchor);

  llvm::DenseSet<compiler::HardwareEmissionOccurrenceId> occurrences;
  for (const auto &module : plan->get().modules)
    for (const auto &child : module.children)
      EXPECT_TRUE(occurrences.insert(child.id).second);
  for (const auto &definition : plan->get().definitions)
    for (const auto &child : definition.templateChildren)
      EXPECT_TRUE(occurrences.insert(child.id).second);
  EXPECT_EQ(occurrences.size(), plan->get().occurrenceAnchors.size());
}

TEST_F(HardwareEmissionPlanTest, RejectsMutatedGenericTargetInventories) {
  auto package =
      parse("tests/compiler/lit/CodeGen/Inputs/generic-forward.mlir");
  ASSERT_TRUE(package);
  Builder builder(&context);
  (*package)->setAttr("ac.stage", builder.getStringAttr("final"));

  auto prepared = compiler::prepareHardwareEmissionPlan(*package);
  ASSERT_TRUE(succeeded(prepared));
  const auto &plan = prepared->get();
  auto identity = llvm::find_if(plan.definitions, [](const auto &item) {
    return item.symbol == "identity";
  });
  auto top = llvm::find_if(
      plan.definitions, [](const auto &item) { return item.symbol == "top"; });
  ASSERT_NE(identity, plan.definitions.end());
  ASSERT_NE(top, plan.definitions.end());
  ASSERT_FALSE(top->templateSchedule.empty());

  auto includingUnit = llvm::find_if(plan.sourceUnits, [&](const auto &unit) {
    return llvm::is_contained(unit.cppIncludes, identity->names.header);
  });
  ASSERT_NE(includingUnit, plan.sourceUnits.end());
  const unsigned unitIndex = includingUnit - plan.sourceUnits.begin();
  compiler::HardwareEmissionPlan mutated = plan;
  auto include = llvm::find(mutated.sourceUnits[unitIndex].cppIncludes,
                            identity->names.header);
  ASSERT_NE(include, mutated.sourceUnits[unitIndex].cppIncludes.end());
  mutated.sourceUnits[unitIndex].cppIncludes.erase(include);
  EXPECT_TRUE(failed(compiler::verifyHardwareEmissionPlan(mutated)));

  mutated = plan;
  auto mutatedIdentity =
      llvm::find_if(mutated.definitions,
                    [](const auto &item) { return item.symbol == "identity"; });
  ASSERT_NE(mutatedIdentity, mutated.definitions.end());
  mutatedIdentity->templateSchedule.push_back(top->templateSchedule.front());
  EXPECT_TRUE(failed(compiler::verifyHardwareEmissionPlan(mutated)));
}

TEST_F(HardwareEmissionPlanTest, RejectsMutatedFrozenIdentities) {
  auto package =
      parse("tests/compiler/lit/CodeGen/Inputs/sampled-admission.mlir");
  ASSERT_TRUE(package);
  auto prepared = compiler::prepareHardwareEmissionPlan(*package);
  ASSERT_TRUE(succeeded(prepared));

  compiler::HardwareEmissionPlan mutated = prepared->get();
  ASSERT_FALSE(mutated.modules.empty());
  ASSERT_FALSE(mutated.modules.front().dependencies.empty());
  mutated.modules.front().dependencies.front().output.id =
      mutated.modules.front().values.size();
  EXPECT_TRUE(failed(compiler::verifyHardwareEmissionPlan(mutated)));

  mutated = prepared->get();
  ASSERT_FALSE(mutated.modules.front().types.empty());
  mutated.modules.front().types.front().id =
      std::numeric_limits<compiler::HardwareEmissionTypeQueryId>::max();
  EXPECT_TRUE(failed(compiler::verifyHardwareEmissionPlan(mutated)));

  mutated = prepared->get();
  ASSERT_FALSE(mutated.occurrenceAnchors.empty());
  mutated.occurrenceAnchors.pop_back();
  EXPECT_TRUE(failed(compiler::verifyHardwareEmissionPlan(mutated)));
}

TEST_F(HardwareEmissionPlanTest, ReportsFrozenPhysicalProjectionRejection) {
  auto package =
      parse("tests/compiler/lit/CodeGen/Inputs/sampled-admission.mlir");
  ASSERT_TRUE(package);
  auto prepared = compiler::prepareHardwareEmissionPlan(*package);
  ASSERT_TRUE(succeeded(prepared));

  std::string diagnostics;
  ScopedDiagnosticHandler handler(
      prepared->get().package->getContext(), [&](Diagnostic &diagnostic) {
        llvm::raw_string_ostream out(diagnostics);
        diagnostic.print(out);
        return success();
      });
  EXPECT_TRUE(failed(compiler::prepareVerilogRenderPlan(*prepared, true)));
  EXPECT_NE(diagnostics.find("physical RTL projection is unavailable"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("physical RTL requires"), std::string::npos)
      << diagnostics;
}

TEST_F(HardwareEmissionPlanTest, RejectsExplicitNonFinalPackages) {
  auto package = parse("tests/compiler/lit/CodeGen/Inputs/two-leaves.mlir");
  ASSERT_TRUE(package);
  Builder builder(&context);
  (*package)->setAttr("ac.stage", builder.getStringAttr("source"));
  EXPECT_TRUE(failed(compiler::prepareHardwareEmissionPlan(*package)));
}

} // namespace
