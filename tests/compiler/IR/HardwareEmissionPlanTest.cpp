#include "Compiler/HardwareEmissionPlan.h"
#include "Compiler/BackendRenderPlan.h"

#include "mlir/Parser/Parser.h"
#include "pycircuit/Dialect/ACIR/ACIRDialect.h"

#include "llvm/ADT/SmallString.h"
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
    for (const auto &step : module.schedule) {
      EXPECT_NE(step.diagnosticAnchor, nullptr);
      EXPECT_FALSE(step.operation.empty());
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
  ASSERT_EQ(root.children.size(), 2u);
  ASSERT_TRUE(root.children[0].moduleIndex);
  ASSERT_TRUE(root.children[1].moduleIndex);
  EXPECT_NE(*root.children[0].moduleIndex, *root.children[1].moduleIndex);
  const auto &scalar = plan->get().modules[*root.children[0].moduleIndex];
  const auto &aggregate = plan->get().modules[*root.children[1].moduleIndex];
  ASSERT_EQ(scalar.key.types.size(), 1u);
  ASSERT_EQ(aggregate.key.types.size(), 1u);
  EXPECT_NE(scalar.key.types.front().value, aggregate.key.types.front().value);
}

TEST_F(HardwareEmissionPlanTest, RejectsExplicitNonFinalPackages) {
  auto package = parse("tests/compiler/lit/CodeGen/Inputs/two-leaves.mlir");
  ASSERT_TRUE(package);
  Builder builder(&context);
  (*package)->setAttr("ac.stage", builder.getStringAttr("source"));
  EXPECT_TRUE(failed(compiler::prepareHardwareEmissionPlan(*package)));
}

} // namespace
