add_acir_library(ACIRSourceCompiler SOURCES
  AnchorArtifact.cpp
  ExecutionProjection.cpp
  FixedUnsignedDivRemLowering.cpp
  NumericLowering.cpp
  PythonImportAST.cpp
  PythonImportCaseSyntax.cpp
  PythonImportContext.cpp
  PythonImportEnumSyntax.cpp
  PythonImportRecords.cpp
  SourceHeaderRegistry.cpp
  SourceHeaderHelpers.cpp
  SourceBodySnapshots.cpp
  SourceDependencies.cpp
  SourceDeclarationContext.cpp
  SourceEntryBoundary.cpp
  SourceLink.cpp
  SourceNamespace.cpp
  SourceMemoryBindings.cpp
  SourceRuleWrites.cpp
  SourceUnit.cpp
  TableQueryLowering.cpp
  TableQueryStages.cpp
  LINK_LIBRARIES
    ACIRDialect
    ACIRAnalysis
    GFIRValidation
    HWIRValidation
    ACIRToGFIR
    GFIRToHWIR
    ACIRTransforms
    MLIRArithDialect
    MLIRFuncDialect
    MLIRParser
    MLIRPass
    MLIRSCFDialect
)
target_include_directories(ACIRSourceCompiler PUBLIC
  "$<BUILD_INTERFACE:${PYCIRCUIT_SOURCE_DIR}/lib>"
  "$<INSTALL_INTERFACE:include>"
)

add_acir_library(ACIRBackendRender SOURCES
  BackendRender.cpp
  LINK_LIBRARIES ACIRDialect
)
target_include_directories(ACIRBackendRender PUBLIC
  "$<BUILD_INTERFACE:${PYCIRCUIT_SOURCE_DIR}/lib>"
  "$<INSTALL_INTERFACE:include>"
)

add_acir_library(ACIRCodegen SOURCES
  BackendRenderPlan.cpp
  FinalCppNames.cpp
  FinalEmitCpp.cpp
  FinalEmitVerilog.cpp
  FinalSourceMap.cpp
  HardwareEmissionPlan.cpp
  HardwareEmitCommon.cpp
  HardwareEmitCppChecks.cpp
  HardwareEmitCppDefinition.cpp
  HardwareEmitVerilogFaults.cpp
  HardwareEmitVerilogDefinition.cpp
  LINK_LIBRARIES ACIRBackendRender ACIRDialect ACIRAnalysis HWIRValidation MLIRParser
)
target_include_directories(ACIRCodegen PUBLIC
  "$<BUILD_INTERFACE:${PYCIRCUIT_SOURCE_DIR}/lib>"
  "$<INSTALL_INTERFACE:include>"
)
