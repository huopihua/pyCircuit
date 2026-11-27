    SmallVector<std::pair<uint64_t, Value>, 0> queuePrefixProofs;
    for (auto [ordinal, proposal] : llvm::enumerate(preparation.queueProposals)) {
      const auto &access = plan.queueAccesses[proposal.access];
      auto request = yield->getOperand(proposal.pathResult).getDefiningOp<ac::BitsBinaryOp>();
      if (!request || request.getOpcode() != "and" ||
          request.getRhs() != record.queuePaths[proposal.access])
        return reject("original queue request raw path differs");
      Value complete = request.getLhs();
      if (failed(binaryProof(request.getResult(), complete, record.queuePaths[proposal.access],
                              "and", access.operation->getLoc()))) return failure();
      if (proposal.excludedControls.empty()) {
        if (complete != prefix) return reject("original queue complete prefix differs");
        continue;
      }
      bool shared = false;
      for (auto [previous, expected] : queuePrefixProofs) {
        if (!charge(budget, access.operation, 1 + proposal.excludedControls.size())) return failure();
        if (preparation.queueProposals[previous].excludedControls != proposal.excludedControls) continue;
        if (complete != expected) return reject("original queue shared retained-control prefix differs");
        shared = true;
        break;
      }
      if (shared) continue;
      if (failed(retainedPrefixProof(complete, proposal.excludedControls,
                                      preparation.controls.size(), access.operation)))
        return failure();
      if (!prepayAppend(budget, access.operation, queuePrefixProofs)) return failure();
      queuePrefixProofs.push_back({ordinal, complete});
    }
