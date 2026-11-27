    for (const auto &proposal : preparation.memoryProposals) {
      const auto &access = plan.memoryAccesses[proposal.access];
      auto request = yield->getOperand(proposal.pathResult).getDefiningOp<ac::BitsBinaryOp>();
      if (!request || request.getOpcode() != "and" ||
          request.getRhs() != record.memoryPaths[proposal.access] ||
          failed(binaryProof(request.getResult(), request.getLhs(), request.getRhs(),
                              "and", access.operation->getLoc())))
        return reject("original Memory demand raw path differs");
      if (access.kind == ac::TransactionMemoryAccessKind::Read) {
        if (proposal.reachedControlEnd < access.arguments.size())
          return failure();
        Value complete = request.getLhs();
        uint64_t first =
            proposal.reachedControlEnd - access.arguments.size();
        for (uint64_t index = proposal.reachedControlEnd; index > first;
             --index) {
          auto joined = complete.getDefiningOp<ac::BitsBinaryOp>();
          if (!joined || joined.getOpcode() != "and" ||
              joined.getRhs() != record.controls[index - 1].accepted ||
              failed(binaryProof(joined.getResult(), joined.getLhs(),
                                 joined.getRhs(), "and",
                                 access.operation->getLoc())))
            return reject("original Memory address witness prefix differs");
          complete = joined.getLhs();
        }
        if (failed(literalProof(
                complete, complete.getType(),
                plan.transaction->getAttrOfType<DictionaryAttr>("occurrence"),
                access.operation->getLoc(), true, &rule.getBody().front())))
          return failure();
      } else if (request.getLhs() != prefix)
        return reject("original Memory issue lacks final atomic admission");
    }
