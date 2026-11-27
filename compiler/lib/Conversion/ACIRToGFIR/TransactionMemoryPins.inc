// Memory contributes ordinary typed pin proposals. All proposals retain their
// resource owner and physical operand identity while crossing existing Modules.
class MemoryPinMaterializer {
public:
  MemoryPinMaterializer(const PreparedTransactionLowering &prepared,
                        MaterializedTransactionLowering &candidate,
                        SourceCheckBudget &budget)
      : prepared(prepared), candidate(candidate), budget(budget),
        hardware(prepared.original), routes(prepared, candidate.packageMap) {}
  LogicalResult run() {
    for (const auto &definition : prepared.definitions) {
      if (definition.memoryIncoming.empty()) continue;
      auto module = cast<ac::ModuleOp>(candidate.packageMap.lookup(definition.module.operator->()));
      Builder b(module.getContext());
      auto signature = module.getFunctionType();
      SmallVector<Type> inputs(signature.getInputs());
      SmallVector<Attribute> names(module.getInputNames().getValue()), defaults;
      if (auto existing = module->getAttrOfType<DictionaryAttr>("ac.internal_ports"))
        llvm::append_range(defaults, existing.getAs<ArrayAttr>("root_inputs"));
      for (auto [ordinal, route] : llvm::enumerate(definition.memoryIncoming)) {
        auto add = [&](Type type, StringRef suffix) {
          std::string name = (Twine("__ac_memory_") + suffix + "_" + Twine(ordinal)).str();
          while (llvm::is_contained(names, b.getStringAttr(name)) ||
                 llvm::is_contained(module.getOutputNames().getValue(), b.getStringAttr(name))) name += "_";
          names.push_back(b.getStringAttr(name)); inputs.push_back(type);
          defaults.push_back(b.getI64IntegerAttr(0));
          return module.getBody().front().addArgument(type, module.getLoc());
        };
        if (!charge(budget, module, 32)) return failure();
        Value path = add(route.predicateType, "path");
        Value data = add(route.type, "data");
        incoming[definition.module].push_back({path, data});
      }
      module->setAttr("function_type", TypeAttr::get(FunctionType::get(module.getContext(), inputs, signature.getResults())));
      module->setAttr("input_names", b.getArrayAttr(names));
      module->setAttr("ac.internal_ports", b.getDictionaryAttr({
          b.getNamedAttr("source_inputs", b.getI64IntegerAttr(definition.sourceInputs)),
          b.getNamedAttr("source_outputs", b.getI64IntegerAttr(definition.sourceOutputs)),
          b.getNamedAttr("root_inputs", b.getArrayAttr(defaults))}));
    }
    for (auto [index, transaction] : llvm::enumerate(prepared.transactions)) {
      const auto &plan = *transaction.plan;
      auto rule = candidate.transactions[index].rule;
      for (const auto &proposal : transaction.memoryProposals) {
        if (proposal.physicalReuse) continue;
        const auto &access = plan.memoryAccesses[proposal.access];
        const auto &owner = plan.memoryOwners[access.owner];
        auto operands = memoryOperands(access, owner.layout);
        unsigned argument = 0;
        for (unsigned operand : operands) {
          Value data;
          bool enable = operand == owner.layout.writeEnableOperand && access.kind == ac::TransactionMemoryAccessKind::Write;
          enable |= access.kind == ac::TransactionMemoryAccessKind::Request && operand == owner.layout.readEnableOperands[access.port];
          if (!enable) data = rule->getResult(proposal.argumentResults[argument++]);
          local[plan.routingModule][key(anchorPath(plan, owner.anchor), owner.allocation, operand)]
              .push_back({rule->getResult(proposal.pathResult), data, access.operation});
        }
        if (argument != proposal.argumentResults.size()) return failure();
      }
    }
    for (const auto &parent : prepared.definitions) {
      for (auto original : parent.module->getRegion(0).front().getOps<ac::InstanceOp>()) {
        auto child = dyn_cast_or_null<ac::ModuleOp>(hardware.resolveCallee(original));
        if (child) {
          const auto &definition = prepared.definitions[prepared.definitionIndex.lookup(child)];
          auto instance = cast<ac::InstanceOp>(candidate.packageMap.lookup(original.operator->()));
          SmallVector<Value> operands(instance.getInputs());
          OpBuilder builder(instance);
          for (const auto &route : definition.memoryIncoming) {
            SmallVector<ac::InstanceOp> path{original}; llvm::append_range(path, route.path);
            auto pins = pinsFor(parent.module, path, route.allocation, route.operand,
                                instance->getLoc(), builder);
            if (failed(pins)) return failure();
            operands.push_back(pins->first); operands.push_back(pins->second);
          }
          instance->setOperands(operands);
        } else if (original->hasAttr("ac.source_memory")) {
          auto memory = cast<ac::InstanceOp>(candidate.packageMap.lookup(original.operator->()));
          OpBuilder builder(memory);
          // Clock/reset are source-domain pins. Every functional pin is either
          // a proved atomic proposal or its safe idle carrier.
          for (unsigned operand = 2; operand < memory.getNumOperands(); ++operand) {
            auto pins = pinsFor(parent.module, {}, original, operand, original.getLoc(), builder);
            if (failed(pins)) return failure();
            memory->setOperand(operand, pins->second);
          }
          memory->removeAttr("ac.source_memory");
          memory->removeAttr("ac.payload_constraint");
        }
      }
    }
    for (auto [channelId, channel] :
         llvm::enumerate(prepared.memoryReadChannels)) {
      if (channelId >= candidate.memoryReadInstances.size())
        return failure();
      auto memory = candidate.memoryReadInstances[channelId];
      auto original = channel.allocation;
      OpBuilder builder(memory);
      for (unsigned operand = 0; operand < memory.getNumOperands(); ++operand) {
        if (operand == 2)
          continue;
        auto canonical = cast<ac::InstanceOp>(
            candidate.packageMap.lookup(original.operator->()));
        memory->setOperand(operand, canonical.getOperand(operand));
      }
      auto transaction = prepared.transactions[channel.transaction];
      auto proposal = llvm::find_if(
          transaction.memoryProposals, [&](const auto &candidate) {
            return candidate.access == channel.access;
          });
      if (proposal == transaction.memoryProposals.end() ||
          !proposal->readChannel || *proposal->readChannel != channelId)
        return failure();
      auto rule = candidate.transactions[channel.transaction].rule;
      Value address = rule->getResult(proposal->argumentResults.front());
      Value path = rule->getResult(proposal->pathResult);
      if (channel.path.empty()) {
        candidate.memoryReadInputs.push_back({path, address});
      } else {
        Value routedPath = path, routedAddress = address;
        for (auto edge : channel.path) {
          auto allocation = cast<ac::InstanceOp>(
              candidate.packageMap.lookup(edge.operator->()));
          auto child = cast<ac::ModuleOp>(hardware.resolveCallee(edge));
          const auto &routing = prepared.definitions[
              prepared.definitionIndex.lookup(child.operator->())];
          auto found = routing.memoryReadOutputIndex.find(channelId);
          if (found == routing.memoryReadOutputIndex.end())
            return failure();
          uint64_t base = routing.sourceInputs + 2 * routing.downward.size() +
                          2 * found->second;
          Value oldPath = allocation->getOperand(base);
          Value oldAddress = allocation->getOperand(base + 1);
          allocation->setOperand(base, routedPath);
          allocation->setOperand(base + 1, routedAddress);
          if (auto result = dyn_cast<OpResult>(oldPath);
              result && result.use_empty())
            result.getOwner()->erase();
          if (auto result = dyn_cast<OpResult>(oldAddress);
              result && result.use_empty())
            result.getOwner()->erase();
          auto module = cast<ac::ModuleOp>(
              candidate.packageMap.lookup(child.operator->()));
          routedPath = module.getBody().front().getArgument(base);
          routedAddress = module.getBody().front().getArgument(base + 1);
        }
      }
      auto owner = original->getParentOfType<ac::ModuleOp>();
      const auto &ownerRouting = prepared.definitions[
          prepared.definitionIndex.lookup(owner.operator->())];
      auto ownerRoute = ownerRouting.memoryReadOutputIndex.find(channelId);
      if (ownerRoute != ownerRouting.memoryReadOutputIndex.end()) {
        uint64_t base = ownerRouting.sourceInputs +
                        2 * ownerRouting.downward.size() +
                        2 * ownerRoute->second;
        auto module = cast<ac::ModuleOp>(
            candidate.packageMap.lookup(owner.operator->()));
        path = module.getBody().front().getArgument(base);
        address = module.getBody().front().getArgument(base + 1);
      }
      Value zero = bitsLiteral(address.getType(), original.getOccurrence(),
                               original.getLoc(), false, builder);
      ac::HardwareBindings bindings;
      bindings.owner = original->getParentOfType<ac::ModuleOp>();
      auto selected = materializeSelection(
          original, bindings, path, address, zero,
          address.getType(), builder, budget);
      if (failed(selected))
        return failure();
      memory->setOperand(2, selected->value);
      if (candidate.memoryReadSelections.size() != channelId)
        return failure();
      candidate.memoryReadSelections.push_back(std::move(*selected));
    }
    return success();
  }
private:
  struct Proposal { Value path, data; Operation *source; };
  QueueRoutingKey key(ArrayRef<ac::InstanceOp> path, ac::InstanceOp memory, unsigned operand) {
    return {{routePath(prepared.original->getContext(), path), memory.getOutputs()[0]}, operand};
  }
  FailureOr<std::pair<Value, Value>> pinsFor(ac::ModuleOp module,
      ArrayRef<ac::InstanceOp> path, ac::InstanceOp memory, unsigned operand,
      Location location, OpBuilder &builder) {
    auto identity = key(path, memory, operand);
    if (auto found = merged[module].find(identity); found != merged[module].end()) return found->second;
    const auto &definition = prepared.definitions[prepared.definitionIndex.lookup(module)];
    ac::HardwareBindings bindings; bindings.owner = module;
    for (auto edge : path) {
      auto child = hardware.bindInstance(edge, bindings);
      if (failed(child)) return failure();
      bindings = *child;
    }
    auto type = hardware.resolveType(memory.getInputs()[operand].getType(), bindings, module);
    auto bit = hardware.resolveType(memory.getInputs()[0].getType(), bindings, module);
    if (failed(type) || failed(bit) || !charge(budget, module, 32)) return failure();
    Value active = bitsLiteral(*bit, memory.getOccurrence(), location, false, builder);
    Value data;
    if (isa<ac::BitsType>(*type)) {
      data = bitsLiteral(*type, memory.getOccurrence(), location, false, builder);
    } else {
      // A nominal type need not contain a zero encoding. Retain the original
      // typed raw carrier for an idle data pin; its enable remains false.
      auto idle = routes.value(module, path, memory.getInputs()[operand]);
      if (failed(idle)) return failure();
      data = *idle;
    }
    detail::MaterializedMemoryPin record;
    record.module = module; record.key = identity; record.idle = data;
    SmallVector<Proposal> proposals;
    if (auto incomingRoute = definition.memoryIncomingIndex.find(identity);
        incomingRoute != definition.memoryIncomingIndex.end()) {
      auto pair = incoming[module][incomingRoute->second];
      proposals.push_back({pair.first, pair.second, memory});
    }
    if (auto found = local[module].find(identity); found != local[module].end())
      llvm::append_range(proposals, found->second);
    for (const auto &proposal : proposals) {
      if (!charge(budget, proposal.source, 32)) return failure();
      Value payload = proposal.data;
      if (!payload) payload = bitsLiteral(*type, memory.getOccurrence(), location, true, builder);
      auto selected = materializeSelection(memory, bindings, proposal.path,
                                            payload, data, *type, builder, budget);
      if (failed(selected))
        return memory.emitOpError()
               << "failed to materialize routed Memory pin selection for "
               << "operand " << operand;
      if (!prepayAppend(budget, memory, record.selections)) return failure();
      data = selected->value;
      record.selections.push_back(std::move(*selected));
      OperationState join(location, ac::BitsBinaryOp::getOperationName());
      join.addOperands({active, proposal.path}); join.addTypes(*bit);
      join.addAttribute("opcode", builder.getStringAttr("or"));
      active = builder.create(join)->getResult(0);
    }
    std::pair<Value, Value> result{active, data};
    if (!prepayAppend(budget, memory, candidate.memoryPins)) return failure();
    record.active = active; record.data = data;
    candidate.memoryPins.push_back(std::move(record));
    merged[module].try_emplace(identity, result);
    return result;
  }
  const PreparedTransactionLowering &prepared;
  MaterializedTransactionLowering &candidate;
  SourceCheckBudget &budget;
  ac::HardwareAnalysis hardware;
  MaterializationRoutes routes;
  llvm::DenseMap<Operation *, SmallVector<std::pair<Value, Value>, 0>> incoming;
  llvm::DenseMap<Operation *, llvm::DenseMap<QueueRoutingKey, SmallVector<Proposal, 0>>> local;
  llvm::DenseMap<Operation *, llvm::DenseMap<QueueRoutingKey, std::pair<Value, Value>>> merged;
};
