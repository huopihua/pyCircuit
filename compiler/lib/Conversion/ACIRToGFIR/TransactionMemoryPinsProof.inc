  LogicalResult memoryPinsProof() {
    using PinKey = detail::QueueRoutingKey;
    struct ExpectedPin {
      ac::ModuleOp module;
      SmallVector<ac::InstanceOp> path;
      ac::InstanceOp memory;
      unsigned operand;
      SmallVector<std::pair<Value, Value>, 0> local;
    };
    SmallVector<ExpectedPin, 0> expected;
    llvm::DenseMap<Operation *, llvm::DenseMap<PinKey, unsigned>> index;
    auto key = [&](ArrayRef<ac::InstanceOp> path, ac::InstanceOp memory, unsigned operand) -> PinKey {
      return {{routePath(prepared.original->getContext(), path), memory.getOutputs()[0]}, operand};
    };
    auto demand = [&](ac::ModuleOp module, ArrayRef<ac::InstanceOp> path,
                      ac::InstanceOp memory, unsigned operand) -> FailureOr<unsigned> {
      auto identity = key(path, memory, operand);
      auto found = index[module].find(identity);
      if (found != index[module].end()) return found->second;
      if (!charge(budget, memory, 32 + path.size())) return failure();
      unsigned id = expected.size();
      index[module].try_emplace(identity, id);
      expected.push_back({module, SmallVector<ac::InstanceOp>(path), memory, operand, {}});
      return id;
    };
    // Derive private route inventory from original source accesses, including
    // actual lexical edges and physical pin roles, rather than candidate rows.
    llvm::DenseMap<Operation *, SmallVector<PinKey, 0>> routes;
    for (auto registration : prepared.original->getRegion(0).front().getOps<ac::ModuleOp>()) {
      const auto &routing = prepared.definitions[prepared.definitionIndex.lookup(registration)];
      for (uint64_t occurrenceIndex : routing.transactionIndices) {
        auto transactionIndex = originalTransactionIndex(occurrenceIndex);
        if (failed(transactionIndex)) return failure();
        const auto &transaction = prepared.transactions[*transactionIndex];
        const auto &plan = *transaction.plan;
        if (plan.routingModule != registration) return reject("Memory transaction routing owner differs");
        uint64_t proposalOrdinal = 0;
        for (auto [accessId, access] : llvm::enumerate(plan.memoryAccesses)) {
          if (access.kind == ac::TransactionMemoryAccessKind::Result) continue;
          if (proposalOrdinal >= transaction.memoryProposals.size()) return failure();
          const auto &proposal = transaction.memoryProposals[proposalOrdinal++];
          if (proposal.access != accessId) return reject("original Memory proposal source order differs");
          // Original source replay independently validates the reuse witness.
          // A covered Read contributes no additional physical select or OR.
          if (proposal.physicalReuse) continue;
          const auto &owner = plan.memoryOwners[access.owner];
          auto path = anchorPath(plan, owner.anchor);
          auto operands = memoryOperands(access, owner.layout);
          unsigned argument = 0;
          auto rule = candidate.transactions[*transactionIndex].rule;
          for (unsigned operand : operands) {
            auto local = demand(plan.routingModule, path, owner.allocation, operand);
            if (failed(local)) return failure();
            bool enable = access.kind == ac::TransactionMemoryAccessKind::Write && operand == owner.layout.writeEnableOperand;
            enable |= access.kind == ac::TransactionMemoryAccessKind::Request && operand == owner.layout.readEnableOperands[access.port];
            Value data = enable ? Value() : rule->getResult(proposal.argumentResults[argument++]);
            expected[*local].local.push_back({rule->getResult(proposal.pathResult), data});
          }
          for (unsigned offset = 0; offset < path.size(); ++offset) {
            auto child = dyn_cast_or_null<ac::ModuleOp>(hardware.resolveCallee(path[offset]));
            if (!child) return failure();
            auto suffix = ArrayRef<ac::InstanceOp>(path).drop_front(offset + 1);
            for (unsigned operand : operands) {
              auto identity = key(suffix, owner.allocation, operand);
              auto &inventory = routes[child];
              if (!llvm::is_contained(inventory, identity)) inventory.push_back(identity);
            }
          }
        }
        if (proposalOrdinal != transaction.memoryProposals.size()) return reject("original Memory proposal inventory differs");
      }
    }
    for (const auto &definition : prepared.definitions) {
      const auto &inventory = routes[definition.module];
      if (inventory.size() != definition.memoryIncoming.size()) return reject("original Memory private route inventory differs");
      for (auto [ordinal, route] : llvm::enumerate(definition.memoryIncoming)) {
        auto identity = key(route.path, route.allocation, route.operand);
        if (identity != inventory[ordinal] || definition.memoryIncomingIndex.lookup(identity) != ordinal)
          return reject("original Memory private route identity/order differs");
      }
      for (auto original : definition.module->getRegion(0).front().getOps<ac::InstanceOp>()) {
        auto child = dyn_cast_or_null<ac::ModuleOp>(hardware.resolveCallee(original));
        if (child) {
          const auto &callee = prepared.definitions[prepared.definitionIndex.lookup(child)];
          for (const auto &route : callee.memoryIncoming) {
            SmallVector<ac::InstanceOp> path{original}; llvm::append_range(path, route.path);
            if (failed(demand(definition.module, path, route.allocation, route.operand))) return failure();
          }
        } else if (original->hasAttr("ac.source_memory")) {
          for (unsigned operand = 2; operand < original.getNumOperands(); ++operand)
            if (failed(demand(definition.module, {}, original, operand))) return failure();
        }
      }
    }
    if (expected.size() != candidate.memoryPins.size()) return reject("original Memory pin merge inventory differs");
    llvm::DenseSet<std::pair<Operation *, PinKey>> seen;
    for (const auto &actual : candidate.memoryPins) {
      auto foundModule = index.find(actual.module);
      if (foundModule == index.end()) return reject("original Memory pin module differs");
      auto found = foundModule->second.find(actual.key);
      if (found == foundModule->second.end() || !seen.insert({actual.module, actual.key}).second)
        return reject("original Memory pin identity differs or repeats");
      const auto &pin = expected[found->second];
      const auto &definition = prepared.definitions[prepared.definitionIndex.lookup(pin.module)];
      auto module = cast<ac::ModuleOp>(candidate.packageMap.lookup(pin.module.operator->()));
      auto memory = pin.memory;
      Location location = pin.path.empty() ? memory.getLoc() : pin.path.front()->getLoc();
      ac::HardwareBindings bindings; bindings.owner = pin.module;
      for (auto edge : pin.path) {
        auto child = hardware.bindInstance(edge, bindings);
        if (failed(child)) return failure();
        bindings = *child;
      }
      auto type = hardware.resolveType(memory.getInputs()[pin.operand].getType(), bindings, module);
      auto bit = hardware.resolveType(memory.getInputs()[0].getType(), bindings, module);
      if (failed(type) || failed(bit)) return failure();
      SmallVector<std::pair<Value, Value>, 0> contributors;
      auto incoming = definition.memoryIncomingIndex.find(actual.key);
      if (incoming != definition.memoryIncomingIndex.end()) {
        uint64_t base = definition.sourceInputs + 2 * definition.downward.size() +
                        2 * definition.memoryReadOutputs.size() +
                        queueInputCount(definition) + 2 * incoming->second;
        contributors.push_back({module.getBody().front().getArgument(base), module.getBody().front().getArgument(base + 1)});
      }
      llvm::append_range(contributors, pin.local);
      if (actual.selections.size() != contributors.size())
        return reject("original Memory typed pin selection inventory differs");
      Value active = actual.active;
      Builder b(module.getContext());
      for (auto [path, payload] : llvm::reverse(contributors)) {
        (void)payload;
        auto joined = active.getDefiningOp<ac::BitsBinaryOp>();
        if (!joined || joined.getRhs() != path ||
            failed(generatedProof(joined, ac::BitsBinaryOp::getOperationName(),
                {joined.getLhs(), path}, {*bit}, b.getDictionaryAttr({b.getNamedAttr("opcode", b.getStringAttr("or"))}), location, &module.getBody().front())))
          return reject("original Memory pin proposal path union differs");
        active = joined.getLhs();
      }
      if (failed(literalProof(active, *bit, memory.getOccurrence(), location, false, &module.getBody().front()))) return failure();
      Value data = actual.idle;
      if (isa<ac::BitsType>(*type)) {
        if (failed(literalProof(data, *type, memory.getOccurrence(), location, false, &module.getBody().front()))) return failure();
      } else {
        auto idle = routed(pin.module, pin.path, memory.getInputs()[pin.operand]);
        if (failed(idle) || data != *idle)
          return reject("original Memory nominal idle carrier differs");
      }
      // Bindings are local to this physical pin. A cache keyed by their
      // address must not survive into the next pin's reconstructed bindings.
      BranchTypeAdmissionCache pinTypes;
      Operation *last = nullptr;
      for (auto [ordinal, contribution] : llvm::enumerate(contributors)) {
        auto [path, payload] = contribution;
        const auto &selection = actual.selections[ordinal];
        if (!payload) {
          auto selected = selection.value ? selection.value.getDefiningOp<ac::BitsSelectOp>() : ac::BitsSelectOp();
          if (!selected || !operations.contains(selected.operator->()) || !isa<ac::BitsType>(*type))
            return reject("original Memory issue enable carrier differs");
          payload = selected.getTrueValue();
          if (failed(literalProof(payload, *type, memory.getOccurrence(), location, true, &module.getBody().front()))) return failure();
        }
        if (failed(localBranchProof(memory, *type, bindings, path, payload, data,
                                    selection, &module.getBody().front(), last, &pinTypes))) return failure();
        data = selection.value;
      }
      if (data != actual.data) return reject("original Memory typed pin merge result differs");
      if (pin.path.empty()) {
        auto allocation = cast<ac::InstanceOp>(candidate.packageMap.lookup(memory.operator->()));
        if (allocation.getInputs()[pin.operand] != actual.data) return reject("original Memory physical pin differs");
      } else {
        auto allocation = cast<ac::InstanceOp>(candidate.packageMap.lookup(pin.path.front().operator->()));
        auto child = dyn_cast<ac::ModuleOp>(hardware.resolveCallee(pin.path.front()));
        const auto &callee = prepared.definitions[prepared.definitionIndex.lookup(child)];
        auto route = callee.memoryIncomingIndex.find(key(ArrayRef<ac::InstanceOp>(pin.path).drop_front(), memory, pin.operand));
        if (route == callee.memoryIncomingIndex.end()) return failure();
        uint64_t base = callee.sourceInputs + 2 * callee.downward.size() +
                        2 * callee.memoryReadOutputs.size() +
                        queueInputCount(callee) + 2 * route->second;
        if (allocation.getInputs()[base] != actual.active || allocation.getInputs()[base + 1] != actual.data)
          return reject("original Memory hierarchical pin pair differs");
      }
    }
    // Every occurrence binds the definition-level asynchronous read inventory.
    // Only edges on a channel's exact lexical path carry its path/address
    // delta; all other occurrences retain the typed zero pair created when the
    // shared definition suffix was installed.
    for (const auto &parent : prepared.definitions) {
      for (auto original :
           parent.module->getRegion(0).front().getOps<ac::InstanceOp>()) {
        auto child =
            dyn_cast_or_null<ac::ModuleOp>(hardware.resolveCallee(original));
        if (!child) continue;
        const auto &routing = prepared.definitions[
            prepared.definitionIndex.lookup(child.operator->())];
        if (routing.memoryReadOutputs.empty()) continue;
        auto allocation = dyn_cast_or_null<ac::InstanceOp>(
            candidate.packageMap.lookupOrNull(original.operator->()));
        auto bindings = hardware.bindInstance(original);
        if (!allocation || failed(bindings))
          return reject("original Memory read occurrence binding is absent");
        for (auto [ordinal, route] :
             llvm::enumerate(routing.memoryReadOutputs)) {
          if (route.channel >= prepared.memoryReadChannels.size())
            return reject("original Memory read occurrence channel differs");
          const auto &channel = prepared.memoryReadChannels[route.channel];
          bool driven = false;
          for (auto [offset, edge] : llvm::enumerate(channel.path)) {
            if (!charge(budget, original)) return failure();
            if (edge == original &&
                ArrayRef<ac::InstanceOp>(channel.path).drop_front(offset + 1) ==
                    ArrayRef<ac::InstanceOp>(route.path)) {
              driven = true;
              break;
            }
          }
          if (driven) continue;
          uint64_t base = routing.sourceInputs +
                          2 * routing.downward.size() + 2 * ordinal;
          if (base + 1 >= allocation.getNumOperands())
            return reject("original Memory read occurrence suffix is absent");
          auto pathType = hardware.resolveType(route.predicateType, *bindings,
                                               original);
          auto addressType = hardware.resolveType(route.addressType, *bindings,
                                                  original);
          if (failed(pathType) || failed(addressType) ||
              failed(literalProof(allocation.getOperand(base), *pathType,
                                  original.getOccurrence(), original.getLoc(),
                                  false, allocation->getBlock())) ||
              failed(literalProof(allocation.getOperand(base + 1),
                                  *addressType, original.getOccurrence(),
                                  original.getLoc(), false,
                                  allocation->getBlock())))
            return reject("original Memory read occurrence idle pair differs");
        }
      }
    }
    if (candidate.memoryReadInstances.size() !=
            prepared.memoryReadChannels.size() ||
        candidate.memoryReadSelections.size() !=
            prepared.memoryReadChannels.size())
      return reject("original Memory read channel inventory differs");
    for (auto [channelId, channel] :
         llvm::enumerate(prepared.memoryReadChannels)) {
      auto original = channel.allocation;
      auto canonical = dyn_cast_or_null<ac::InstanceOp>(
          candidate.packageMap.lookupOrNull(original.operator->()));
      auto instance = candidate.memoryReadInstances[channelId];
      if (!canonical || !instance || !operations.contains(instance.operator->()) ||
          instance->getBlock() != canonical->getBlock() ||
          instance->getLoc() != original.getLoc() ||
          instance->getResultTypes() != canonical->getResultTypes() ||
          instance->getNumOperands() != canonical->getNumOperands())
        return reject("original Memory read channel allocation differs");
      NamedAttrList expected(canonical->getAttrs());
      expected.set(
          "instance_name",
          StringAttr::get(instance.getContext(),
                          (Twine(canonical.getInstanceName()) + "__ac_read_" +
                           Twine(channelId))
                              .str()));
      auto occurrence = canonical.getOccurrence();
      auto site = occurrence.getAs<DictionaryAttr>("site");
      SmallVector<Attribute> astPath(site.getAs<ArrayAttr>("ast_path").begin(),
                                     site.getAs<ArrayAttr>("ast_path").end());
      Builder b(instance.getContext());
      astPath.push_back(b.getDictionaryAttr(
          {b.getNamedAttr("kind", b.getStringAttr("index")),
           b.getNamedAttr("value", b.getI64IntegerAttr(channelId))}));
      NamedAttrList channelSite(site);
      channelSite.set("ast_path", b.getArrayAttr(astPath));
      expected.set(
          "occurrence",
          b.getDictionaryAttr(
              {b.getNamedAttr("site", channelSite.getDictionary(b.getContext())),
               b.getNamedAttr("expansion",
                              occurrence.getAs<ArrayAttr>("expansion"))}));
      expected.erase("ac.source_memory");
      expected.erase("ac.payload_constraint");
      expected.set("ac.memory_read_mirror", canonical.getOccurrence());
      auto canonicalProperties =
          cast<DictionaryAttr>(canonical->getPropertiesAsAttribute());
      NamedAttrList expectedProperties(canonicalProperties);
      expectedProperties.set("instance_name", expected.get("instance_name"));
      expectedProperties.set("occurrence", expected.get("occurrence"));
      if (instance->getAttrs() != expected.getAttrs() ||
          instance->getPropertiesAsAttribute() !=
              expectedProperties.getDictionary(instance.getContext()) ||
          failed(cover(instance)))
        return reject("original Memory read channel identity differs");
      for (unsigned operand = 0; operand < instance.getNumOperands(); ++operand)
        if (operand != 2 &&
            instance->getOperand(operand) != canonical->getOperand(operand))
          return reject("original Memory read channel shared pin differs");
      const auto &transaction = prepared.transactions[channel.transaction];
      auto proposal = llvm::find_if(
          transaction.memoryProposals, [&](const auto &candidate) {
            return candidate.access == channel.access;
          });
      if (proposal == transaction.memoryProposals.end() ||
          !proposal->readChannel || *proposal->readChannel != channelId ||
          proposal->argumentResults.empty())
        return reject("original Memory read channel proposal differs");
      auto rule = candidate.transactions[channel.transaction].rule;
      Value sourceAddress = rule->getResult(proposal->argumentResults.front());
      Value sourcePath = rule->getResult(proposal->pathResult);
      Value address = sourceAddress;
      Value path = sourcePath;
      if (channel.path.empty()) {
        auto found = llvm::find(candidate.memoryReadInputs,
                                std::make_pair(sourcePath, sourceAddress));
        if (found == candidate.memoryReadInputs.end())
          return reject("original Memory read channel input route is absent");
      } else {
        auto child = dyn_cast_or_null<ac::ModuleOp>(
            hardware.resolveCallee(channel.path.front()));
        auto allocation = dyn_cast_or_null<ac::InstanceOp>(
            candidate.packageMap.lookupOrNull(
                channel.path.front().operator->()));
        if (!child || !allocation)
          return reject("original Memory read channel input route is absent");
        const auto &routing = prepared.definitions[
            prepared.definitionIndex.lookup(child.operator->())];
        auto found = routing.memoryReadOutputIndex.find(channelId);
        if (found == routing.memoryReadOutputIndex.end())
          return reject("original Memory read channel input route is absent");
        uint64_t base = routing.sourceInputs + 2 * routing.downward.size() +
                        2 * found->second;
        if (base + 1 >= allocation.getNumOperands() ||
            allocation.getOperand(base) != sourcePath ||
            allocation.getOperand(base + 1) != sourceAddress)
          return reject("original Memory read channel input route differs");
        auto owner = original->getParentOfType<ac::ModuleOp>();
        const auto &ownerRouting = prepared.definitions[
            prepared.definitionIndex.lookup(owner.operator->())];
        auto ownerRoute = ownerRouting.memoryReadOutputIndex.find(channelId);
        if (ownerRoute != ownerRouting.memoryReadOutputIndex.end()) {
          uint64_t argument = ownerRouting.sourceInputs +
                              2 * ownerRouting.downward.size() +
                              2 * ownerRoute->second;
          auto module = cast<ac::ModuleOp>(
              candidate.packageMap.lookup(owner.operator->()));
          if (argument + 1 >= module.getBody().front().getNumArguments())
            return reject("original Memory read channel input route differs");
          path = module.getBody().front().getArgument(argument);
          address = module.getBody().front().getArgument(argument + 1);
        }
      }
      const auto &selection = candidate.memoryReadSelections[channelId];
      Value zero;
      if (auto selected = selection.value
                              ? selection.value.getDefiningOp<ac::BitsSelectOp>()
                              : ac::BitsSelectOp()) {
        if (selected.getCondition() != path ||
            selected.getTrueValue() != address)
          return reject("original Memory read channel selection differs");
        zero = selected.getFalseValue();
      } else if (auto merge =
                     selection.value
                         ? selection.value.getDefiningOp<ac::ValueMergeOp>()
                         : ac::ValueMergeOp()) {
        if (merge.getGuards().size() != 1 || merge.getValues().size() != 1 ||
            merge.getGuards()[0] != path || merge.getValues()[0] != address)
          return reject("original Memory read channel selection differs");
        zero = merge.getBase();
      } else {
        return reject("original Memory read channel selection differs");
      }
      if (failed(literalProof(zero, address.getType(),
                                       original.getOccurrence(),
                                       original.getLoc(), false,
                                       instance->getBlock())))
        return reject("original Memory read channel idle address differs");
      ac::HardwareBindings bindings;
      bindings.owner = original->getParentOfType<ac::ModuleOp>();
      Operation *last = nullptr;
      BranchTypeAdmissionCache channelTypes;
      if (failed(localBranchProof(original, address.getType(), bindings, path,
                                  address, zero, selection,
                                  instance->getBlock(), last,
                                  &channelTypes)) ||
          instance->getOperand(2) != selection.value)
        return reject("original Memory read channel address differs");
    }
    return success();
  }
