// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/CVSplit.cpp
 * \brief TileLangIR Cube/Vector 拆分 pass。
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
      DenseMap<Operation *, std::size_t> opGroupId;
      llvm::SmallPtrSet<Operation *, 8> sharedAllocOps;

      // 第一步：只把与 workspace 相连的 copy 作为 C/V 划分种子。
      // workspace 是 DSL 暴露出来的跨 Cube/Vector 边界，因此这些 copy
      // 是当前 pass 判断 scope 边界最稳定的入口。
      SmallVector<CopyOpInterface> workspaceCopySeeds;
      for (auto copyOp : body->getOps<CopyOpInterface>()) {
        if (isValFromWorkspace(copyOp.getSource()) ||
            isValFromWorkspace(copyOp.getTarget()))
          workspaceCopySeeds.push_back(copyOp);
      }

      // 第二步：发现跨种子共享的本地 memref。
      // discovery DFS 不生成 group，只判断某个 local alloc 是否会被多个
      // workspace-copy seed 的依赖树触达；这类 alloc 不能被搬进任一 scope，
      // 否则 sibling scope 会出现 dominance 问题。
      DenseMap<Operation *, Operation *> allocOwnerSeed;
      for (auto copyOp : workspaceCopySeeds) {
        llvm::SmallPtrSet<Operation *, 32> visited;
        llvm::SmallPtrSet<Operation *, 8> expandedScopeUnits;
        discoverSharedAllocs(copyOp.getOperation(), copyOp.getOperation(),
                             allocOwnerSeed, sharedAllocOps, visited,
                             expandedScopeUnits);
        discoverCopyEndpointUsers(copyOp, allocOwnerSeed, sharedAllocOps,
                                  visited, expandedScopeUnits);
      }

      // 第三步：以每个 workspace copy 为根真正收集 group。
      // 收集 DFS 使用和 discovery 相同的数据边/词法边，但遇到上一步标记的
      // shared alloc 会停止，让 shared alloc 留在父循环中支配所有 scope。
      for (auto copyOp : workspaceCopySeeds) {
        LDBG("Start from " << *copyOp.getOperation());
        llvm::SmallPtrSet<Operation *, 32> visited;
        llvm::SmallPtrSet<Operation *, 8> expandedScopeUnits;
        bool groupHasOps = false;
        visitGroupOfOps(copyOp.getOperation(), groupId, opGroupId,
                        sharedAllocOps, visited, expandedScopeUnits,
                        groupHasOps);
        if (groupHasOps)
          groupId++;
      }

      // 第四步：按顶层 op 的 groupId 打包 scope。
      // nested region 内的 op 不单独移动，前面的 DFS 会把其顶层 scf.for/if
      // 作为 scope unit 记录在 opGroupId 里，移动时整体搬迁。
      SmallVector<hivm::TCoreType> coreType(groupId,
                                            hivm::TCoreType::CUBE_OR_VECTOR);
      SmallVector<SmallVector<Operation *>> groups(groupId);
      SmallVector<std::vector<Mapper>> groupResults(groupId);
      for (auto &op : body->getOperations()) {
        if (sharedAllocOps.contains(&op))
          continue;
        if (opGroupId.find(&op) == opGroupId.end())
          continue;
        const auto id = opGroupId[&op];
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

      // 第五步：创建 scope.scope，将同组顶层 op 原序搬入，并把跨 scope 的
      // scf.yield 结果改接到 scope result。
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
        for (auto op : group) {
          op->moveBefore(&scopeBody, scopeBody.end());
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

  // 旧路径用 multi_buffer mark 表示某些本地 alloc 需要留在父循环中。
  // 这里继续把它作为 DFS 边界处理，避免和历史 FA_mix 写法冲突。
  bool touchesMarkedLocalBoundary(Operation *op) {
    for (auto *user : op->getUsers()) {
      auto markOp = dyn_cast<annotation::MarkOp>(user);
      if (markOp && markOp->hasAttr("hivm.multi_buffer")) {
        return true;
      }
    }
    return false;
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

  // 记录 discovery DFS 触达的 local alloc 归属。
  // 返回 true 表示这是该 alloc 第一次被某个 seed 发现，可以继续扩展
  // nested 捕获关系；返回 false 表示已经见过，若来自其他 seed 则标为共享边界。
  bool recordDiscoveredAlloc(
      Operation *allocOp, Operation *seedOp,
      DenseMap<Operation *, Operation *> &allocOwnerSeed,
      llvm::SmallPtrSetImpl<Operation *> &sharedAllocOps) {
    auto owner = allocOwnerSeed.find(allocOp);
    if (owner == allocOwnerSeed.end()) {
      allocOwnerSeed[allocOp] = seedOp;
      return true;
    }
    if (owner->second != seedOp) {
      sharedAllocOps.insert(allocOp);
      return false;
    }
    return false;
  }

  // discovery 阶段遇到 local alloc 后默认不走普通 use-chain，否则会通过
  // UB/L1 复用把整个 loop 泛洪成一个 group。唯一例外是 nested region 捕获：
  // 如果 alloc 被 scf.for/scf.if 内部 op 使用，需要进入该 nested region，
  // 才能发现 mask_ub 与 kv_ub 这类只在词法 scope 中并列出现的关系。
  void discoverNestedUsersOfAlloc(
      Operation *allocOp, Operation *seedOp,
      DenseMap<Operation *, Operation *> &allocOwnerSeed,
      llvm::SmallPtrSetImpl<Operation *> &sharedAllocOps,
      llvm::SmallPtrSetImpl<Operation *> &visited,
      llvm::SmallPtrSetImpl<Operation *> &expandedScopeUnits) {
    for (auto *user : allocOp->getUsers()) {
      Operation *scopeUnit = getTopLevelOpInCurrentFor(user);
      if (!scopeUnit || scopeUnit == user)
        continue;
      discoverSharedAllocs(user, seedOp, allocOwnerSeed, sharedAllocOps,
                           visited, expandedScopeUnits);
    }
  }

  // 把嵌套 region 当成词法边展开。
  // 只要 DFS 触达 region 内任一 op，该 region 的其他 op 也属于同一个
  // 词法 scope；移动阶段仍只移动顶层 scopeUnit，避免拆散 scf.for/scf.if。
  void discoverNestedOps(
      Operation *scopeUnit, Operation *seedOp,
      DenseMap<Operation *, Operation *> &allocOwnerSeed,
      llvm::SmallPtrSetImpl<Operation *> &sharedAllocOps,
      llvm::SmallPtrSetImpl<Operation *> &visited,
      llvm::SmallPtrSetImpl<Operation *> &expandedScopeUnits) {
    if (scopeUnit->getNumRegions() == 0)
      return;
    if (!expandedScopeUnits.insert(scopeUnit).second)
      return;

    scopeUnit->walk([&](Operation *nestedOp) {
      if (nestedOp == scopeUnit)
        return;
      discoverSharedAllocs(nestedOp, seedOp, allocOwnerSeed, sharedAllocOps,
                           visited, expandedScopeUnits);
    });
  }

  // 从 copy 本地端补充一段受限 use-chain。
  // 因为 HIVM 多数计算通过 outs(memref) 写结果，producer/consumer 之间
  // 不一定有 SSA result 边，只沿 def-chain 会找不到真实计算 op。
  void discoverUsersOfCopyEndpoint(
      Value val, Operation *seedOp, bool beforeSeed, bool afterSeed,
      DenseMap<Operation *, Operation *> &allocOwnerSeed,
      llvm::SmallPtrSetImpl<Operation *> &sharedAllocOps,
      llvm::SmallPtrSetImpl<Operation *> &visited,
      llvm::SmallPtrSetImpl<Operation *> &expandedScopeUnits) {
    Operation *allocOp = getLocalAllocRoot(val);
    if (!allocOp)
      return;

    for (auto *user : allocOp->getUsers()) {
      Operation *scopeUnit = getTopLevelOpInCurrentFor(user);
      if (!scopeUnit)
        continue;
      bool inRange = scopeUnit == seedOp;
      if (beforeSeed && scopeUnit->isBeforeInBlock(seedOp))
        inRange = true;
      if (afterSeed && seedOp->isBeforeInBlock(scopeUnit))
        inRange = true;
      if (!inRange)
        continue;
      discoverSharedAllocs(user, seedOp, allocOwnerSeed, sharedAllocOps,
                           visited, expandedScopeUnits);
    }
  }

  // 针对 workspace copy 的方向决定补哪一侧 use-chain：
  // - 本地->工作区：本地端的 producer 通常在 copy 之前；
  // - 工作区->本地：本地端的 consumer 通常在 copy 之后。
  void discoverCopyEndpointUsers(
      CopyOpInterface copyOp,
      DenseMap<Operation *, Operation *> &allocOwnerSeed,
      llvm::SmallPtrSetImpl<Operation *> &sharedAllocOps,
      llvm::SmallPtrSetImpl<Operation *> &visited,
      llvm::SmallPtrSetImpl<Operation *> &expandedScopeUnits) {
    Operation *seedOp = copyOp.getOperation();
    // 本地->工作区的本地端通常由种子 copy 之前的 op 写入；
    // 工作区->本地的本地端通常由种子 copy 之后的 op 消费。
    // HIVM 的结果多写在 outs(memref) 里，因此发现阶段必须按 copy
    // 方向补充这一段受限 use 边，否则找不到生产者/消费者。
    if (isValFromWorkspace(copyOp.getTarget()))
      discoverUsersOfCopyEndpoint(copyOp.getSource(), seedOp,
                                  /*beforeSeed=*/true, /*afterSeed=*/false,
                                  allocOwnerSeed, sharedAllocOps, visited,
                                  expandedScopeUnits);
    if (isValFromWorkspace(copyOp.getSource()))
      discoverUsersOfCopyEndpoint(copyOp.getTarget(), seedOp,
                                  /*beforeSeed=*/false, /*afterSeed=*/true,
                                  allocOwnerSeed, sharedAllocOps, visited,
                                  expandedScopeUnits);
  }

  // discovery DFS：只负责标记跨 seed 共享的 local alloc，不负责生成 group。
  // 遇到 local alloc 后记录归属并停止普通 use-chain；遇到普通 op 时走
  // use/def 数据边，并额外展开 nested region 的词法边。
  void discoverSharedAllocs(
      Operation *op, Operation *seedOp,
      DenseMap<Operation *, Operation *> &allocOwnerSeed,
      llvm::SmallPtrSetImpl<Operation *> &sharedAllocOps,
      llvm::SmallPtrSetImpl<Operation *> &visited,
      llvm::SmallPtrSetImpl<Operation *> &expandedScopeUnits) {
    Operation *scopeUnit = getTopLevelOpInCurrentFor(op);
    if (!scopeUnit)
      return;

    if (isa<bishengir::memref_ext::AllocWorkspaceOp>(op))
      return;
    if (isa<memref::AllocOp>(op)) {
      bool firstSeen = recordDiscoveredAlloc(op, seedOp, allocOwnerSeed,
                                             sharedAllocOps);
      // 发现阶段遇到本地 memref 后不沿普通 use-chain 泛洪；
      // 但如果该 memref 被嵌套 region 捕获，需要进入这个 region，
      // 否则 gather 里的 mask_ub / kv_ub 这种词法同 scope 关系会漏掉。
      if (firstSeen)
        discoverNestedUsersOfAlloc(op, seedOp, allocOwnerSeed, sharedAllocOps,
                                   visited, expandedScopeUnits);
      return;
    }

    if (!visited.insert(op).second)
      return;
    if (touchesMarkedLocalBoundary(scopeUnit))
      return;

    discoverNestedOps(scopeUnit, seedOp, allocOwnerSeed, sharedAllocOps,
                      visited, expandedScopeUnits);

    for (auto *user : op->getUsers()) {
      if (!currentForOp->isProperAncestor(user))
        continue;
      if (isa<scf::YieldOp>(user))
        continue;
      discoverSharedAllocs(user, seedOp, allocOwnerSeed, sharedAllocOps,
                           visited, expandedScopeUnits);
    }

    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp)
        continue;
      if (isScalarOp(definingOp) ||
          isa<bishengir::memref_ext::AllocWorkspaceOp>(definingOp))
        continue;
      discoverSharedAllocs(definingOp, seedOp, allocOwnerSeed, sharedAllocOps,
                           visited, expandedScopeUnits);
    }
  }

  // collection DFS 中的 nested region 展开。
  // 触达 region 内任一 op 后，将 region 内其他 op 一并纳入搜索；但最终
  // opGroupId 记录的是顶层 scopeUnit，因此移动时不会破坏嵌套结构。
  void visitNestedOps(Operation *scopeUnit, std::size_t groupId,
                      DenseMap<Operation *, std::size_t> &opGroupId,
                      llvm::SmallPtrSetImpl<Operation *> &sharedAllocOps,
                      llvm::SmallPtrSetImpl<Operation *> &visited,
                      llvm::SmallPtrSetImpl<Operation *> &expandedScopeUnits,
                      bool &groupHasOps) {
    if (scopeUnit->getNumRegions() == 0)
      return;
    if (!expandedScopeUnits.insert(scopeUnit).second)
      return;

    scopeUnit->walk([&](Operation *nestedOp) {
      if (nestedOp == scopeUnit)
        return;
      visitGroupOfOps(nestedOp, groupId, opGroupId, sharedAllocOps, visited,
                      expandedScopeUnits, groupHasOps);
    });
  }

  // collection DFS：把当前 seed 依赖树上的顶层可移动 op 写入 opGroupId。
  // currentForOp 不是按“op 类型是否为 for”来做退出条件，而是限定本轮
  // CVSplit 只打包当前 loop body 里能整体移动的直接子 op。嵌套 region
  // 里的 op 会先映射到这个直接子 op，再按词法 scope 展开。
  // 其他停止条件是：遇到 shared alloc 边界、遇到历史 mark 边界，或遇到
  // 已经属于其他 group 的非 alloc op。若 local alloc 被多个 group 竞争，
  // 则降级为 shared alloc，留在父循环中支配所有使用点。
  void visitGroupOfOps(Operation *op, std::size_t groupId,
                       DenseMap<Operation *, std::size_t> &opGroupId,
                       llvm::SmallPtrSetImpl<Operation *> &sharedAllocOps,
                       llvm::SmallPtrSetImpl<Operation *> &visited,
                       llvm::SmallPtrSetImpl<Operation *> &expandedScopeUnits,
                       bool &groupHasOps) {
    Operation *scopeUnit = getTopLevelOpInCurrentFor(op);
    if (!scopeUnit)
      return;
    if (sharedAllocOps.contains(scopeUnit))
      return;
    if (touchesMarkedLocalBoundary(scopeUnit))
      return;

    auto it = opGroupId.find(scopeUnit);
    if (it == opGroupId.end()) {
      opGroupId[scopeUnit] = groupId;
      groupHasOps = true;
    } else if (it->second != groupId) {
      if (isa<memref::AllocOp>(scopeUnit)) {
        sharedAllocOps.insert(scopeUnit);
        opGroupId.erase(scopeUnit);
      }
      return;
    }

    if (!visited.insert(op).second)
      return;

    // 嵌套 region 是一个词法 scope：只要其中任意 op 被收集，
    // 外层 region op 和 region 内其他 op 都要作为同一个 scope 单元处理。
    visitNestedOps(scopeUnit, groupId, opGroupId, sharedAllocOps, visited,
                   expandedScopeUnits, groupHasOps);

    LDBG("Visiting " << *op);

    for (auto user : op->getUsers()) {
      if (!currentForOp->isProperAncestor(user))
        continue;
      if (isa<scf::YieldOp>(user))
        continue;
      visitGroupOfOps(user, groupId, opGroupId, sharedAllocOps, visited,
                      expandedScopeUnits, groupHasOps);
    }

    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp)
        continue;
      if (isScalarOp(definingOp) ||
          isa<bishengir::memref_ext::AllocWorkspaceOp>(definingOp))
        continue;
      visitGroupOfOps(definingOp, groupId, opGroupId, sharedAllocOps, visited,
                      expandedScopeUnits, groupHasOps);
    }
  }

  // 将任意 op 映射为 currentForOp body 的直接子 op。
  // 如果 op 在嵌套 scf.for/scf.if 内部，这里返回外层那个直接子 op；
  // 如果 op 不属于 currentForOp 的 region，则返回 nullptr，表示它不是
  // 本轮 scope 打包可移动的对象，应留给外层/内层 loop 自己处理或保持外部支配。
  Operation *getTopLevelOpInCurrentFor(Operation *op) {
    auto *body = currentForOp.getBody();
    Operation *cur = op;
    while (cur && cur->getBlock() != body)
      cur = cur->getParentOp();
    return cur;
  }

  // 判断一个 value 是否来自 workspace alloc；subview/reinterpret_cast
  // 这类 view op 会递归追溯到原始 source。
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

  // 纯 scalar op 不携带 tensor/memref/vector 数据依赖，不作为 C/V scope
  // 扩张依据，避免把索引计算等无关标量链路扩得过大。
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
