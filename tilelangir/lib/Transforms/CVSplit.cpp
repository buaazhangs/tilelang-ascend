// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/CVSplit.cpp
 * \brief TileLangIR CV split pass.
 *
 */

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "tilelangir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tilelangir {

#define GEN_PASS_DEF_TILELANGIRCVSPLIT
#include "tilelangir/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tilelangir-cv-split"
#define LDBG(X)                                                                \
  LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] " << X << '\n')

struct TileLangIRCVSplit : impl::TileLangIRCVSplitBase<TileLangIRCVSplit> {

  struct Mapper {
    OpResult def;
    OpOperand &use;

    Mapper(OpResult def, OpOperand &use) : def(def), use(use) {}
  };

  void runOnOperation() override {
    getOperation().walk([this](scf::ForOp forOp) {
      LDBG("Processing " << forOp);
      currentForOp = forOp;
      auto body = forOp.getBody();

      std::size_t groupId = 0;
      DenseMap<const Operation *, std::size_t> opGroupId;
      llvm::SmallPtrSet<Operation *, 8> sharedAllocOps;

      SmallVector<CopyOpInterface> workspaceCopySeeds;
      for (auto copyOp : body->getOps<CopyOpInterface>()) {
        if (isValFromWorkspace(copyOp.getSource()) ||
            isValFromWorkspace(copyOp.getTarget()))
          workspaceCopySeeds.push_back(copyOp);
      }

      // First identify local buffers reached from more than one workspace copy
      // seed. Discovery intentionally follows only def chains from the
      // non-workspace endpoint. Following memref.alloc users here would flood
      // through reused UB storage and over-mark the graph.
      DenseMap<Operation *, Operation *> allocOwnerSeed;
      for (auto copyOp : workspaceCopySeeds) {
        llvm::SmallPtrSet<Operation *, 8> seedAllocs;
        collectLocalAllocsFromCopy(copyOp, seedAllocs);
        for (Operation *allocOp : seedAllocs) {
          auto owner = allocOwnerSeed.find(allocOp);
          if (owner == allocOwnerSeed.end()) {
            allocOwnerSeed[allocOp] = copyOp.getOperation();
          } else if (owner->second != copyOp.getOperation()) {
            sharedAllocOps.insert(allocOp);
          }
        }
      }

      // A shared local alloc has two legal meanings in user DSL:
      //   1. no explicit workspace->local reload in this scope: keep the alloc
      //      in the parent loop and let scopes read the same persistent UB/L1.
      //   2. explicit workspace->local reload in this scope: treat that reload
      //      as creating a private local buffer for this scope.
      //
      // Track only case (2) here. Case (1) remains an external operand because
      // sharedAllocOps are skipped when packing groups.
      DenseMap<Operation *, SmallVector<Operation *>> reloadCopiesByAlloc;
      for (auto copyOp : workspaceCopySeeds) {
        if (!isValFromWorkspace(copyOp.getSource()))
          continue;
        Operation *targetAlloc = getLocalAllocRoot(copyOp.getTarget());
        if (targetAlloc && sharedAllocOps.contains(targetAlloc))
          reloadCopiesByAlloc[targetAlloc].push_back(copyOp.getOperation());
      }

      for (auto copyOp : workspaceCopySeeds) {
        LDBG("Start from " << *copyOp.getOperation());
        llvm::SmallPtrSet<Operation *, 32> visited;
        bool groupHasOps = false;
        Operation *reloadAlloc = nullptr;
        Operation *reloadEnd = nullptr;
        if (isValFromWorkspace(copyOp.getSource())) {
          reloadAlloc = getLocalAllocRoot(copyOp.getTarget());
          if (!reloadAlloc || !sharedAllocOps.contains(reloadAlloc))
            reloadAlloc = nullptr;
          if (reloadAlloc)
            reloadEnd =
                getNextReloadCopy(reloadAlloc, copyOp.getOperation(),
                                  reloadCopiesByAlloc);
        }
        visitGroupOfOps(copyOp, [&](Operation *op) {
                // Reused local buffers must stay in the parent loop so they
                // dominate every C/V scope that reads or writes the same storage.
                // Treat them as def-chain boundaries during grouping; otherwise
                // their use lists can connect otherwise independent seed trees.
                if (sharedAllocOps.contains(op))
                  return true;
                auto it = opGroupId.find(op);
                if (it == opGroupId.end()) {
                  opGroupId[op] = groupId;
                  groupHasOps = true;
                  return false;
                }
                return it->second != groupId;
              },
                        visited, sharedAllocOps, reloadAlloc,
                        copyOp.getOperation(), reloadEnd);
        if (groupHasOps)
          groupId++;
      }

      SmallVector<hivm::TCoreType> coreType(groupId,
                                            hivm::TCoreType::CUBE_OR_VECTOR);
      SmallVector<SmallVector<Operation *>> groups(groupId);
      SmallVector<std::vector<Mapper>> groupResults(groupId);
      SmallVector<SmallVector<Operation *>> groupPrivateAllocs(groupId);
      for (auto &op : body->getOperations()) {
        if (sharedAllocOps.contains(&op))
          continue;
        if (opGroupId.find(&op) == opGroupId.end())
          continue;
        const auto id = opGroupId[&op];
        if (auto copyOp = dyn_cast<CopyOpInterface>(op)) {
          if (isValFromWorkspace(copyOp.getSource())) {
            Operation *targetAlloc = getLocalAllocRoot(copyOp.getTarget());
            if (targetAlloc && sharedAllocOps.contains(targetAlloc))
              addUnique(groupPrivateAllocs[id], targetAlloc);
          }
        }
        auto opCoreType = hivm::TCoreType::CUBE_OR_VECTOR;
        if (auto hivmOp = dyn_cast<hivm::HIVMStructuredOp>(op);
            hivmOp && hivmOp.getCoreType())
          opCoreType = *hivmOp.getCoreType();
        LDBG("Collecting op " << op << " belonging to group " << id
                              << " with core type "
                              << hivm::stringifyTCoreType(opCoreType));

        if (opCoreType != hivm::TCoreType::CUBE_OR_VECTOR)
          switch (coreType[id]) {
          case hivm::TCoreType::CUBE_OR_VECTOR:
            coreType[id] = opCoreType;
            break;
          case hivm::TCoreType::CUBE:
          case hivm::TCoreType::VECTOR:
            coreType[id] = coreType[id] == opCoreType
                               ? opCoreType
                               : hivm::TCoreType::CUBE_AND_VECTOR;
            break;
          }

        groups[id].push_back(&op);

        for (auto result : op.getResults()) {
          for (auto &use : result.getUses()) {
            if (isa<scf::YieldOp>(use.getOwner()))
              groupResults[id].emplace_back(result, use);
          }
        }
      }

      OpBuilder builder(body->getTerminator());
      for (std::size_t id = 0; id < groups.size(); ++id) {
        auto &group = groups[id];
        auto &resultMaps = groupResults[id];
        auto scope = builder.create<scope::ScopeOp>(
            builder.getUnknownLoc(),
            TypeRange(llvm::map_to_vector(resultMaps, [](const Mapper &map) {
              return map.use.get().getType();
            })));
        for (auto &&[resultId, map] : llvm::enumerate(resultMaps)) {
          map.use.assign(scope.getResult(resultId));
        }
        scope->setAttr(hivm::TCoreTypeAttr::name,
                       builder.getAttr<hivm::TCoreTypeAttr>(coreType[id]));

        auto &scopeBody = scope.getRegion().emplaceBlock();
        DenseMap<Value, Value> privateAllocMap;
        OpBuilder allocBuilder(&scopeBody, scopeBody.begin());
        for (Operation *allocOp : groupPrivateAllocs[id]) {
          Operation *clonedAlloc = allocBuilder.clone(*allocOp);
          privateAllocMap[allocOp->getResult(0)] = clonedAlloc->getResult(0);
        }
        for (auto op : group) {
          op->moveBefore(&scopeBody, scopeBody.end());
        }
        if (!privateAllocMap.empty()) {
          scope.walk([&](Operation *op) {
            for (OpOperand &operand : op->getOpOperands()) {
              auto it = privateAllocMap.find(operand.get());
              if (it != privateAllocMap.end())
                operand.set(it->second);
            }
          });
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(&scopeBody);
        builder.create<scope::ReturnOp>(
            builder.getUnknownLoc(),
            ValueRange(llvm::map_to_vector(
                resultMaps, [](Mapper &map) -> Value { return map.def; })));

        LDBG("Packed a " << hivm::stringifyTCoreType(coreType[id])
                         << "-core scope:\n"
                         << scope);
      }
    });
  }

private:
  scf::ForOp currentForOp;

  bool touchesMarkedLocalBoundary(Operation *op) {
    for (auto *user : op->getUsers()) {
      auto markOp = dyn_cast<annotation::MarkOp>(user);
      if (markOp && markOp->hasAttr("hivm.multi_buffer")) {
        return true;
      }
    }
    return false;
  }

  void collectLocalAllocsFromCopy(
      CopyOpInterface copyOp, llvm::SmallPtrSetImpl<Operation *> &allocs) {
    llvm::SmallPtrSet<Operation *, 16> visited;
    for (auto val : {copyOp.getSource(), copyOp.getTarget()}) {
      if (isValFromWorkspace(val))
        continue;
      auto definingOp = val.getDefiningOp();
      if (!definingOp)
        continue;
      collectLocalAllocsFromDefChain(definingOp, allocs, visited);
    }
  }

  void collectLocalAllocsFromDefChain(
      Operation *op, llvm::SmallPtrSetImpl<Operation *> &allocs,
      llvm::SmallPtrSetImpl<Operation *> &visited) {
    if (!getTopLevelOpInCurrentFor(op))
      return;
    if (!visited.insert(op).second)
      return;
    if (isa<bishengir::memref_ext::AllocWorkspaceOp>(op))
      return;
    if (isa<memref::AllocOp>(op)) {
      allocs.insert(op);
      return;
    }

    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp)
        continue;
      if (isScalarOp(definingOp))
        continue;
      collectLocalAllocsFromDefChain(definingOp, allocs, visited);
    }
  }

  Operation *getLocalAllocRoot(Value val) {
    auto definingOp = val.getDefiningOp();
    if (!definingOp)
      return nullptr;
    if (isa<memref::AllocOp>(definingOp))
      return definingOp;
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(definingOp))
      return getLocalAllocRoot(viewOp.getViewSource());
    return nullptr;
  }

  Operation *getNextReloadCopy(
      Operation *allocOp, Operation *start,
      DenseMap<Operation *, SmallVector<Operation *>> &reloadCopiesByAlloc) {
    Operation *next = nullptr;
    auto reloadIt = reloadCopiesByAlloc.find(allocOp);
    if (reloadIt == reloadCopiesByAlloc.end())
      return nullptr;
    for (Operation *candidate : reloadIt->second) {
      if (candidate == start)
        continue;
      if (!start->isBeforeInBlock(candidate))
        continue;
      if (!next || candidate->isBeforeInBlock(next))
        next = candidate;
    }
    return next;
  }

  bool isInReloadEpoch(Operation *op, Operation *start, Operation *end) {
    Operation *scopeUnit = getTopLevelOpInCurrentFor(op);
    if (!scopeUnit)
      return false;
    if (scopeUnit != start && scopeUnit->isBeforeInBlock(start))
      return false;
    if (end && (scopeUnit == end || end->isBeforeInBlock(scopeUnit)))
      return false;
    return true;
  }

  void addUnique(SmallVectorImpl<Operation *> &ops, Operation *op) {
    if (!llvm::is_contained(ops, op))
      ops.push_back(op);
  }

  void visitGroupOfOps(Operation *op,
                       llvm::function_ref<bool(Operation *)> visitor,
                       llvm::SmallPtrSetImpl<Operation *> &visited,
                       llvm::SmallPtrSetImpl<Operation *> &sharedAllocOps,
                       Operation *reloadAlloc = nullptr,
                       Operation *reloadStart = nullptr,
                       Operation *reloadEnd = nullptr) {
    Operation *scopeUnit = getTopLevelOpInCurrentFor(op);
    if (!scopeUnit)
      return;
    if (!visited.insert(scopeUnit).second)
      return;
    if (sharedAllocOps.contains(scopeUnit)) {
      // A shared alloc is a graph boundary: do not add it to any group, and do
      // not let it connect unrelated workspace-copy seed trees. For an explicit
      // reload into this alloc, continue only through users in the reload epoch
      // [reloadStart, nextReload), then the scope packer clones the alloc and
      // remaps those users to the private copy.
      if (scopeUnit != reloadAlloc || !reloadStart)
        return;
      for (auto user : scopeUnit->getUsers()) {
        if (!currentForOp->isProperAncestor(user))
          continue;
        if (isa<scf::YieldOp>(user))
          continue;
        if (!isInReloadEpoch(user, reloadStart, reloadEnd))
          continue;
        visitGroupOfOps(user, visitor, visited, sharedAllocOps, reloadAlloc,
                        reloadStart, reloadEnd);
      }
      return;
    }
    if (touchesMarkedLocalBoundary(scopeUnit))
      return;
    if (visitor(scopeUnit))
      return;
    LDBG("Visiting " << *op);

    for (auto user : op->getUsers()) {
      if (!currentForOp->isProperAncestor(user))
        continue;
      if (isa<scf::YieldOp>(user))
        continue;
      visitGroupOfOps(user, visitor, visited, sharedAllocOps, reloadAlloc,
                      reloadStart, reloadEnd);
    }

    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp)
        continue;
      if (isScalarOp(definingOp) ||
          isa<bishengir::memref_ext::AllocWorkspaceOp>(definingOp))
        continue;
      visitGroupOfOps(definingOp, visitor, visited, sharedAllocOps, reloadAlloc,
                      reloadStart, reloadEnd);
    }
  }

  Operation *getTopLevelOpInCurrentFor(Operation *op) {
    auto *body = currentForOp.getBody();
    Operation *cur = op;
    while (cur && cur->getBlock() != body)
      cur = cur->getParentOp();
    return cur;
  }

  bool isValFromWorkspace(Value val) {
    auto definingOp = val.getDefiningOp();
    return definingOp &&
           TypeSwitch<Operation *, bool>(definingOp)
               .Case([&](ViewLikeOpInterface op) {
                 return isValFromWorkspace(op.getViewSource());
               })
               .Case([](bishengir::memref_ext::AllocWorkspaceOp op) {
                 return true;
               })
               .Default(false);
  }

  bool isScalarOp(Operation *op) {
    auto isScalar = [](Value val) {
      return !isa<TensorType, BaseMemRefType, VectorType>(val.getType());
    };
    return llvm::all_of(op->getOperands(), isScalar) &&
           llvm::all_of(op->getResults(), isScalar);
  }
};
#undef DEBUG_TYPE

} // namespace mlir::tilelangir
