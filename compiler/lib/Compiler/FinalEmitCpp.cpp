#include "FinalEmitCpp.h"
#include "BackendPlanning.h"
#include "CppWorkOutlining.h"
#include "FinalCppNames.h"
#include "HardwareEmitCommon.h"
#include "HardwareEmitCppChecks.h"
#include "pycircuit/Dialect/ACIR/SourceUnitValidation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace acir::compiler {
namespace {
LogicalResult emitEnum(HardwareEmitContext &context, ac::EnumOp declaration,
                       raw_ostream &out) {
  auto type =
      ac::EnumType::get(declaration.getContext(), declaration.getSymNameAttr());
  auto definition = context.resolveEnum(type, declaration);
  auto name = context.cppType(type, declaration);
  if (failed(definition) || failed(name))
    return failure();
  SmallVector<StringRef> parts;
  StringRef(*name).drop_front(2).split(parts, "::");
  for (size_t i = 0; i + 1 < parts.size(); ++i)
    out << "namespace " << parts[i] << " { ";
  out << "struct " << parts.back() << " { gfsim::Bits<" << definition->width
      << "> code{}; };\n";
  for (size_t i = 0; i + 1 < parts.size(); ++i)
    out << "}\n";
  out << "template <> struct gfsim::hardware_traits<" << *name << "> : "
      << "gfsim::hardware_struct_traits<" << *name << ", &" << *name
      << "::code> {};\n";
  return success();
}
LogicalResult emitRecord(HardwareEmitContext &context, ac::StructOp record,
                         raw_ostream &out, llvm::DenseSet<Operation *> &done) {
  if (done.contains(record))
    return success();
  for (auto raw : record.getFields()) {
    auto field = cast<DictionaryAttr>(raw);
    Type dependency = field.getAs<TypeAttr>("type").getValue();
    while (auto table = dyn_cast<ac::TableType>(dependency))
      dependency = table.getElementType();
    if (auto nested = dyn_cast<ac::StructType>(dependency)) {
      auto declaration = dyn_cast_or_null<ac::StructOp>(
          context.lookupDefinition(nested.getName().getValue()));
      if (!declaration || failed(emitRecord(context, declaration, out, done)))
        return failure();
    }
  }
  done.insert(record);
  auto name = context.cppType(
      ac::StructType::get(record.getContext(), record.getSymNameAttr()),
      record);
  if (failed(name))
    return failure();
  StringRef qualified = *name;
  SmallVector<StringRef> parts;
  qualified.drop_front(2).split(parts, "::");
  for (size_t i = 0; i + 1 < parts.size(); ++i)
    out << "namespace " << parts[i] << " { ";
  out << "struct " << parts.back() << " {\n";
  SmallVector<std::string> pointers;
  for (auto raw : record.getFields()) {
    auto field = cast<DictionaryAttr>(raw);
    Type fieldType = field.getAs<TypeAttr>("type").getValue();
    auto type = context.cppType(fieldType, record);
    auto source = field.getAs<StringAttr>("name");
    auto member = legalizeIdentifier(source.getValue(),
                                     [&] { return record.emitOpError(); });
    if (failed(type) || failed(member))
      return failure();
    out << "  " << *type << " " << *member;
    // Nested records default-construct their initialized leaves. Repeating an
    // empty member initializer at every record level is unnecessary.
    if (!isa<ac::StructType>(fieldType))
      out << "{}";
    out << ";\n";
    pointers.push_back("&" + *name + "::" + *member);
  }
  out << "};\n";
  for (size_t i = 0; i + 1 < parts.size(); ++i)
    out << "}\n";
  out << "template <> struct gfsim::hardware_traits<" << *name << "> : "
      << "gfsim::hardware_struct_traits<" << *name;
  for (auto &pointer : pointers)
    out << ", " << pointer;
  out << "> {};\n";
  return success();
}
FailureOr<ac::ModuleOp> rootModule(HardwareEmitContext &context) {
  auto root =
      context.plan.rootModule < context.plan.modules.size()
          ? dyn_cast_or_null<ac::ModuleOp>(
                context.plan.modules[context.plan.rootModule].diagnosticAnchor)
          : ac::ModuleOp();
  if (!root)
    return context.emitPlanError() << "root module is unresolved";
  return root;
}
struct SystemControls {
  std::string clock, reset;
};
FailureOr<SystemControls> systemControls(HardwareEmitContext &context) {
  auto root = context.plan.modules[context.plan.rootModule].diagnosticAnchor;
  auto fail = [&] { return root->emitOpError(); };
  const HardwareEmissionRootInterfacePlan &interface =
      context.plan.rootInterface;
  if (!interface.closedSystem)
    return fail() << "generated simulation requires a verified system root";
  if (!interface.clockInput || !interface.resetInput)
    return fail() << "system root requires verified clock/reset domains";
  unsigned clockOrdinal = *interface.clockInput;
  unsigned resetOrdinal = *interface.resetInput;
  if (clockOrdinal == resetOrdinal || interface.publicInputCount != 2 ||
      clockOrdinal >= interface.publicInputCount ||
      resetOrdinal >= interface.publicInputCount ||
      interface.publicOutputCount != 0)
    return fail() << "system root must be closed except for clock/reset";
  auto legal = [&](uint64_t ordinal) {
    return legalizeIdentifier(interface.inputNames[ordinal], fail);
  };
  auto clockName = legal(clockOrdinal), resetName = legal(resetOrdinal);
  if (failed(clockName) || failed(resetName))
    return failure();
  return SystemControls{std::move(*clockName), std::move(*resetName)};
}
std::string systemSimulationMain(const SystemControls &controls,
                                 StringRef config, ArrayRef<std::string> samples,
                                 bool resetIfOne) {
  std::string text;
  llvm::raw_string_ostream out(text);
  std::string declarations, drive, cycleGuard;
  if (!samples.empty()) {
    declarations = "constexpr std::string_view pyc_samples[] = {";
    for (StringRef row : samples)
      declarations += "\"" + row.str() + "\",";
    declarations +=
        "};\n"
        "auto pyc_symbol(char symbol) {\n"
        "  if (symbol == '0' || symbol == '1') return pyc_bit(symbol == '1');\n"
        "  if (symbol == 'x') return gfsim::wire<gfsim::Bits<1>>::unknown();\n"
        "  return "
        "gfsim::wire<gfsim::Bits<1>>::fromPacked(gfsim::FourState<1>::"
        "highImpedance());\n}\n";
    drive = "    if (epoch >= " + std::to_string(samples.size()) +
            ") return false;\n"
            "    self.inputs." +
            controls.clock +
            " = pyc_symbol(pyc_samples[epoch][0]);\n"
            "    self.inputs." +
            controls.reset + " = pyc_symbol(pyc_samples[epoch][1]);\n";
    cycleGuard = "      std::cerr << \"pycircuit_system: explicit --cycles "
                 "conflicts with samples\\n\"; return 2;\n";
  } else {
    drive = "    if (epoch >= self.cycles * 2) return false;\n"
            "    self.inputs." +
            controls.clock +
            " = pyc_bit((epoch & 1u) != 0);\n"
            "    self.inputs." +
            controls.reset + " = pyc_bit(false);\n";
  }
  out << "#include <gfsim/SystemRunner.h>\n"
         "#include \"pycircuit_system.hpp\"\n"
         "#include <charconv>\n#include <cstdint>\n#include <iostream>\n"
         "#include <limits>\n#include <string>\n#include <string_view>\n"
         "#include <vector>\n\n"
         "namespace {\n"
         "constexpr std::string_view pyc_simulation_config = R\"pyc("
      << config
      << ")pyc\";\n"
         "constexpr std::uint64_t pyc_max_cycles = "
         "4611686018427387903ULL;\n"
         "auto pyc_bit(bool value) {\n"
         "  return gfsim::wire<gfsim::Bits<1>>::known("
         "gfsim::Bits<1>{value ? 1u : 0u});\n}\n"
      << declarations
      << "struct pyc_run_context {\n"
         "  pyc_dut &dut;\n  std::uint64_t cycles;\n"
         "  pyc_dut::Inputs inputs{};\n"
         "  static void initialize(void *opaque) {\n"
         "    auto &self = *static_cast<pyc_run_context *>(opaque);\n"
         "    self.inputs."
      << controls.clock
      << " = pyc_bit(false);\n"
         "    self.inputs."
      << controls.reset
      << " = pyc_bit(false);\n"
         "    self.dut.drive(self.inputs);\n  }\n"
         "  static bool drive(void *opaque, std::uint64_t epoch) {\n"
         "    auto &self = *static_cast<pyc_run_context *>(opaque);\n"
      << drive
      << "    self.dut.drive(self.inputs);\n    return true;\n  }\n"
         "};\n"
         "bool pyc_cycles(std::string_view text, std::uint64_t &value) {\n"
         "  if (text.empty()) return false;\n"
         "  auto result = std::from_chars(text.data(), text.data() + "
         "text.size(), value);\n"
         "  return result.ec == std::errc{} && result.ptr == text.data() + "
         "text.size() && value > 0 && value <= pyc_max_cycles;\n}\n"
         "} // namespace\n\n"
         "int main(int argc, char **argv) {\n"
         "  std::uint64_t cycles = 10;\n  bool has_cycles = false;\n"
         "  bool has_events = false;\n"
         "  std::vector<std::string> forwarded;\n"
         "  forwarded.emplace_back(argv && argc > 0 ? argv[0] : "
         "\"pycircuit_system\");\n"
         "  for (int index = 1; index < argc; ++index) {\n"
         "    std::string_view option = argv[index];\n"
         "    if (option == \"--cycles\") {\n"
      << cycleGuard
      << "      if (has_cycles || ++index >= argc || "
         "!pyc_cycles(argv[index], cycles)) {\n"
         "        std::cerr << \"pycircuit_system: --cycles requires one "
         "positive decimal value\\n\";\n        return 2;\n      }\n"
         "      has_cycles = true;\n      continue;\n    }\n"
         "    has_events = has_events || option == \"--events\";\n"
         "    forwarded.emplace_back(argv[index]);\n  }\n"
         "  if (!has_events) { forwarded.emplace_back(\"--events\"); "
         "forwarded.emplace_back(\"-\"); }\n"
         "  std::vector<char *> runner_args;\n"
         "  for (auto &argument : forwarded) "
         "runner_args.push_back(argument.data());\n"
         "  gfsim::SystemRunner runner(static_cast<int>(runner_args.size()), "
         "runner_args.data(), pyc_simulation_config);\n"
         "  if (!runner.ready()) return 2;\n"
         "  pyc_dut dut(runner.workers());\n"
         "  pyc_run_context context{dut, cycles};\n"
         "  const gfsim::RunnerCallbacks callbacks{&context, "
         "&pyc_run_context::initialize, &pyc_run_context::drive, nullptr, \""
      << (resetIfOne ? "if_one" : "strict") << "\", " << samples.size()
      << "};\n"
         "  return runner.Run(dut.system(), dut.observations(), "
         "pyc_observation_metadata(), callbacks);\n}\n";
  out.flush();
  return text;
}
} // namespace
FailureOr<FinalCppSourceParts>
lowerCppSourceParts(const VerifiedHardwareEmissionPlan &hardware) {
  HardwareEmitContext context(
      hardware.get(), BackendPlanningAccess::sourceChecks(hardware),
      BackendPlanningAccess::observationPlan(hardware), nullptr);
  // Native preparation builds the shared observation and source-check plans.
  // Closed systems and checked module roots consume reachable checks through
  // their generated execution boundary. Checks belonging only to unselected
  // definitions are still verified but do not require a root check lifecycle.
  if (failed(context.prepareNativeChecks()))
    return hardware.get().package->emitOpError()
           << "C++ emission preparation failed";
  auto root = rootModule(context);
  if (failed(root))
    return failure();
  auto parts = emitPreparedCppSourceParts(context);
  if (failed(parts))
    return hardware.get().package->emitOpError()
           << "C++ source generation failed";
  return parts;
}
FailureOr<FinalCppSourceParts>
emitPreparedCppSourceParts(HardwareEmitContext &context) {
  // The retained view never grants authority after mutation. Both public and
  // internal native gates consume this exact verified emission implementation.
  auto root = rootModule(context);
  if (failed(root))
    return failure();
  const HardwareEmissionRootInterfacePlan &rootInterface =
      context.plan.rootInterface;
  FinalCppSourceParts parts;
  auto rootType = context.rootCppType();
  if (failed(rootType))
    return failure();
  parts.rootCppName = *rootType;
  parts.rootHeaderPath = context.names(*root).header;
  // The descriptor table is the compiled form of every source observation
  // occurrence, so it is built before the glue that carries it. An unobserved
  // design keeps the previous empty configuration and byte-identical glue.
  auto descriptorTable = cppObservationDescriptorTable(context);
  if (failed(descriptorTable))
    return failure();
  std::string observationMetadata;
  if (rootInterface.closedSystem) {
    auto generated = cppObservationRunnerMetadata(context);
    if (failed(generated))
      return failure();
    observationMetadata = std::move(*generated);
  }
  auto configure = cppObservationConfigureCall(context);
  if (failed(configure))
    return failure();
  bool observes = !descriptorTable->empty();
  parts.supportHeader =
      "#pragma once\n#include <gfsim/wire.h>\n#include <gfsim/dff.h>\n"
      "#include <gfsim/sampled_clock.h>\n"
      "#include <gfsim/sync_mem.h>\n#include <gfsim/byte_mem.h>\n#include "
      "<gfsim/fifo.h>\n"
      "#include <gfsim/SimModule.h>\n#include <gfsim/SimSystem.h>\n"
      "#include <gfsim/collection.h>\n"
      "#include <gfsim/WorkExecutor.h>\n"
      "#include <memory>\n#include <string>\n#include <string_view>\n"
      "#include <array>\n#include "
      "<algorithm>\n"
      "#include <bit>\n#include <type_traits>\n#include <utility>\n"
      "#include <cstddef>\n#include <cstdint>\n";
  parts.supportHeader += cppWorkRuntimeSupport().str();
  if (observes)
    parts.supportHeader +=
        "#include <gfsim/ObservationSlot.h>\n"
        "// Native observation binding. A generated module reaches the "
        "system-owned\n"
        "// observation slots through this handle, which the DUT binds before "
        "Build().\n"
        "struct pyc_observation_sink {\n"
        "  gfsim::ObservationSlots *slots = nullptr;\n"
        "  const gfsim::SimSystem *system = nullptr;\n"
        "};\n";
  parts.supportHeader +=
      "constexpr std::int64_t pyc_floordiv(std::int64_t a, std::int64_t b) { "
      "return a/b - ((a%b != 0 && (a<0)!=(b<0)) ? 1 : 0); }\n"
      "constexpr std::int64_t pyc_mod(std::int64_t a, std::int64_t b) { auto "
      "r=a%b; return r != 0 && (r<0)!=(b<0) ? r+b : r; }\n"
      "template <std::size_t N> consteval std::uint64_t pyc_mod_words(\n"
      "  std::array<std::uint64_t,N> words, bool negative, std::uint64_t "
      "modulus) {\n"
      "  std::uint64_t remainder=0;\n"
      "  for (std::size_t i=N; i-- > 0;) for (unsigned bit=64; bit-- > 0;)\n"
      "    remainder=(remainder*2+((words[i]>>bit)&1))%modulus;\n"
      "  return negative && remainder ? modulus-remainder : remainder;\n}\n"
      "template <unsigned W> gfsim::Bits<W> pyc_bits(std::int64_t value) "
      "{ return gfsim::Bits<W>{static_cast<std::uint64_t>(value)}; }\n"
      "\n";
  {
    llvm::raw_string_ostream out(parts.nominalDeclarations);
    for (Operation *declaration : context.plan.nominalDeclarations)
      if (auto enumeration = dyn_cast<ac::EnumOp>(declaration);
          enumeration && failed(emitEnum(context, enumeration, out)))
        return failure();
    llvm::DenseSet<Operation *> done;
    for (Operation *operation : context.plan.nominalDeclarations)
      if (auto record = dyn_cast<ac::StructOp>(operation))
        if (failed(emitRecord(context, record, out, done)))
          return failure();
    out.flush();
  }
  parts.supportHeader += parts.nominalDeclarations;
  for (const HardwareEmissionSourceUnitPlan &unit : context.plan.sourceUnits) {
    FinalCppSourceGroup group;
    auto sourceOwner = freezeFinalSourceOwner(unit.sourceOwner, [&]() {
      return context.plan.package->emitOpError();
    });
    if (failed(sourceOwner))
      return failure();
    group.sourceOwner = std::move(*sourceOwner);
    group.headerPath = unit.cppHeaderPath;
    group.sourcePath = unit.cppSourcePath;
    group.header = "#pragma once\n#include \"pycircuit_support.hpp\"\n";
    for (const std::string &include : unit.cppIncludes)
      group.header += "#include \"" + include + "\"\n";
    {
      llvm::raw_string_ostream out(group.header);
      for (unsigned definitionIndex : unit.emissionOrder) {
        auto module = cast<ac::ModuleOp>(
            context.plan.definitions[definitionIndex].diagnosticAnchor);
        if (failed(emitHardwareCppDefinition(context, module, out)))
          return module.emitOpError() << "C++ definition emission failed";
      }
      out.flush();
    }
    if (!group.sourcePath.empty())
      group.source = "#include \"" + group.headerPath + "\"\n";
    parts.sourceGroups.push_back(std::move(group));
  }
  parts.systemHeader = "#pragma once\n#include \"" + parts.rootHeaderPath +
                       "\"\n"
                       "using pyc_root = " +
                       parts.rootCppName + ";\n";
  if (rootInterface.closedSystem)
    parts.systemHeader += "#include <gfsim/SystemRunner.h>\n";
  if (observes)
    parts.systemHeader += *descriptorTable;
  parts.systemHeader += observationMetadata;
  {
    llvm::raw_string_ostream out(parts.systemHeader);
    out << "#include <gfsim/SimSystem.h>\n#include <stdexcept>\n"
           "class pyc_dut final {\npublic:\n";
    SmallVector<std::string> inputs, outputs;
    auto ports = [&](StringRef typeName, ArrayRef<std::string> names,
                     SmallVectorImpl<std::string> &ports) -> LogicalResult {
      out << "  struct " << typeName << " {\n";
      for (StringRef raw : names) {
        auto name = legalizeIdentifier(raw,
                                       [&] { return root->emitOpError(); });
        if (failed(name))
          return failure();
        ports.push_back(*name);
        out << "    decltype(std::declval<pyc_root>()." << *name << ") "
            << *name << ";\n";
      }
      out << "  };\n";
      return success();
    };
    SmallVector<std::string> exposedOutputs(
        rootInterface.outputNames.begin(),
        rootInterface.outputNames.begin() + rootInterface.publicOutputCount);
    for (unsigned companion : rootInterface.availabilityOutputs)
      exposedOutputs.push_back(rootInterface.outputNames[companion]);
    if (failed(ports("Inputs",
                     ArrayRef(rootInterface.inputNames)
                         .take_front(rootInterface.publicInputCount),
                     inputs)) ||
        failed(ports("Outputs", exposedOutputs, outputs)))
      return failure();
    out << "  explicit pyc_dut(std::size_t workers = 1) : executor_(workers), ";
    if (observes)
      out << "root_(std::make_shared<pyc_root>(\"root\", &executor_, "
             "&observation_sink_, 0))";
    else
      out << "root_(std::make_shared<pyc_root>(\"root\", &executor_))";
    if (context.sourceChecks().hasChecks())
      out << ", system_(*root_)";
    out << " {\n";
    if (observes)
      out << "    observation_sink_.slots = &observations_;\n"
             "    observation_sink_.system = &system_;\n";
    out << "    if (!" << *configure
        << " || "
           "!system_.AttachObservations(observations_) || "
           "!system_.AddModule(*root_)) throw std::logic_error(\"DUT setup "
           "failed\");\n"
           "  }\n  pyc_dut(const pyc_dut &) = delete;\n"
           "  pyc_dut &operator=(const pyc_dut &) = delete;\n"
           "  // Control-thread access only, between sampling epochs.\n"
           "  void drive(const Inputs &inputs) {\n";
    for (auto &name : inputs)
      out << "    root_->" << name << " = inputs." << name << ";\n";
    out << "  }\n  Outputs sample() const {\n"
           "    if (system_.state() != gfsim::SimSystemState::Ready || "
           "system_.cycle() == 0) throw std::logic_error(\"no successful DUT "
           "sample\");\n"
           "    return {";
    llvm::interleaveComma(outputs, out, [&](const std::string &name) {
      out << "root_->" << name;
    });
    out << "};\n  }\n  gfsim::SimSystem &system() { return system_; }\n"
           "  gfsim::ObservationSlots &observations() { return observations_; "
           "}\n"
           "private:\n  class System final : public gfsim::SimSystem {\n";
    if (context.sourceChecks().hasChecks())
      out << "  public:\n    explicit System(pyc_root &root) noexcept : "
             "root_(root) {}\n"
             "  private:\n    bool Precheck() noexcept override {\n"
             "      source_failure_ = {};\n"
             "      return root_.pyc_validate_checks(source_failure_);\n    "
             "}\n"
             "    gfsim::SimFailureInfo DescribeFailure(gfsim::SimFailurePhase "
             "phase) "
             "const noexcept override {\n"
             "      if (phase == gfsim::SimFailurePhase::Check && "
             "!source_failure_.code.empty()) return source_failure_;\n"
             "      return gfsim::SimSystem::DescribeFailure(phase);\n    }\n"
             "    pyc_root &root_;\n    gfsim::SimFailureInfo "
             "source_failure_{};\n";
    else
      out << "    bool Precheck() noexcept override { return true; }\n";
    out << "  };\n"
           "  gfsim::WorkExecutor executor_;\n";
    if (observes)
      out << "  pyc_observation_sink observation_sink_;\n";
    out << "  std::shared_ptr<pyc_root> root_;\n"
           "  gfsim::ObservationSlots observations_;\n  System system_;\n};\n";
  }
  if (rootInterface.closedSystem) {
    auto controls = systemControls(context);
    if (failed(controls))
      return failure();
    parts.simulationConfig =
        "{\"deadlock_window\":null,\"max_domain_cycles\":{},"
        "\"max_ticks\":9223372036854775807,"
        "\"schema\":\"pycircuit-model-config\",\"version\":\"1\"}\n";
    parts.simulationMain = systemSimulationMain(
        *controls, StringRef(parts.simulationConfig).trim(),
        context.plan.systemSamples,
        context.resetIfOne());
  }
  return parts;
}
} // namespace acir::compiler
