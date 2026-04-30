// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/CVSplit.cpp
 * \brief TileLangIR Cube/Vector 拆分 pass。
 */

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/IR/HIVMTraits.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "tilelangir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

namespace mlir::tilelangir {

#define GEN_PASS_DEF_TILELANGIRCVSPLIT
#include "tilelangir/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tilelangir-cv-split"
#define LDBG(X)                                                                \
  LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TYPE << "] " << X << '\n')

struct TileLangIRCVSplit : impl::TileLangIRCVSplitBase<TileLangIRCVSplit> {

  struct StageGroup {
    hivm::TCoreType coreType = hivm::TCoreType::CUBE_OR_VECTOR;
    SmallVector<Operation *> ops;
  };

  void runOnOperation() override {
    SmallVector<scf::ForOp> pipelineLoops;
    getOperation().walk([&](scf::ForOp forOp) {
      if (isPipelineFor(forOp))
        pipelineLoops.push_back(forOp);
    });

    for (auto forOp : pipelineLoops) {
      LDBG("Processing pipeline loop " << forOp);
      currentForOp = forOp;

      DenseMap<Operation *, std::size_t> cubeGroupOfOp;
      SmallVector<StageGroup> cubeGroups = collectCubeGroups(cubeGroupOfOp);
      SmallVector<StageGroup> orderedGroups =
          collectVectorGroupsByOrder(cubeGroups, cubeGroupOfOp);

      packGroups(orderedGroups);
    }
  }

private:
  scf::ForOp currentForOp;

  // 当前 CVSplit 只处理 T.pipeline lowering 后带 num_stages 标记的循环。
  // 普通 scf.for 可能只是 gather、mask 或标量控制流，不能被当成 C/V 分割边界。
  bool isPipelineFor(scf::ForOp forOp) {
    return forOp->hasAttr("tilelangir.num_stages");
  }

  // 临时策略：Cube 阶段只以 GEMM/mmadL1 为锚点。workspace copy 是阶段边界，
  // 不是阶段根节点；继续以 copy 为 seed 会把真实计算拆成大量 copy-only scope。
  bool isCubeAnchor(Operation *op) {
    auto name = op->getName().getStringRef();
    return name == "hivm.hir.mmadL1" || name.contains("mmad");
  }

  bool hasOnlyScalarResults(Operation *op) {
    auto isScalar = [](Value val) {
      return !isa<TensorType, BaseMemRefType, VectorType>(val.getType());
    };
    return op->getNumResults() > 0 && llvm::all_of(op->getResults(), isScalar);
  }

  // helper op 没有独立 C/V 含义，并且后续 EnableMultiBuffer 目前不会替换
  // scope result，所以这些 op 先留在父 loop 中支配后续 scope。
  bool isHelperOp(Operation *op) {
    return isa<ViewLikeOpInterface>(op) || isScalarOp(op) ||
           hasOnlyScalarResults(op);
  }

  bool shouldKeepOutsideScope(Operation *op) {
    return isa<memref::AllocOp, bishengir::memref_ext::AllocWorkspaceOp>(op) ||
           isHelperOp(op) || op->getNumResults() != 0;
  }

  bool containsOp(const StageGroup &group, Operation *op) {
    return llvm::any_of(group.ops,
                        [&](Operation *groupOp) { return groupOp == op; });
  }

  void addOpToGroup(StageGroup &group, Operation *op) {
    if (!op || shouldKeepOutsideScope(op) || containsOp(group, op))
      return;
    if (getTopLevelOpInCurrentFor(op) != op)
      return;
    group.ops.push_back(op);
  }

  void sortGroupOps(StageGroup &group) {
    std::sort(group.ops.begin(), group.ops.end(),
              [](Operation *lhs, Operation *rhs) {
                return lhs->isBeforeInBlock(rhs);
              });
  }

  Operation *findWorkspaceToLocalCopyBefore(
      Operation *anchor, Operation *localAlloc,
      const llvm::SmallPtrSetImpl<Operation *> &alreadyInCube) {
    Operation *candidate = nullptr;
    for (auto &op : currentForOp.getBody()->getOperations()) {
      if (&op == anchor)
        break;
      if (alreadyInCube.contains(&op))
        continue;

      auto copyOp = dyn_cast<CopyOpInterface>(&op);
      if (!copyOp)
        continue;
      if (isValFromWorkspace(copyOp.getSource()) &&
          getLocalAllocRoot(copyOp.getTarget()) == localAlloc)
        candidate = &op;
    }
    return candidate;
  }

  Operation *findLocalToWorkspaceCopyAfter(
      Operation *anchor, Operation *localAlloc,
      const llvm::SmallPtrSetImpl<Operation *> &alreadyInCube) {
    bool seenAnchor = false;
    for (auto &op : currentForOp.getBody()->getOperations()) {
      if (&op == anchor) {
        seenAnchor = true;
        continue;
      }
      if (!seenAnchor)
        continue;
      if (isCubeAnchor(&op))
        return nullptr;
      if (alreadyInCube.contains(&op))
        continue;

      auto copyOp = dyn_cast<CopyOpInterface>(&op);
      if (!copyOp)
        continue;
      if (getLocalAllocRoot(copyOp.getSource()) == localAlloc &&
          isValFromWorkspace(copyOp.getTarget()))
        return &op;
    }
    return nullptr;
  }

  void addCubeBoundaryCopies(
      Operation *anchor, StageGroup &group,
      const llvm::SmallPtrSetImpl<Operation *> &alreadyInCube) {
    for (auto operand : anchor->getOperands()) {
      Operation *localAlloc = getLocalAllocRoot(operand);
      if (!localAlloc)
        continue;

      addOpToGroup(group, findWorkspaceToLocalCopyBefore(anchor, localAlloc,
                                                         alreadyInCube));
      addOpToGroup(group, findLocalToWorkspaceCopyAfter(anchor, localAlloc,
                                                       alreadyInCube));
    }
  }

  SmallVector<StageGroup>
  collectCubeGroups(DenseMap<Operation *, std::size_t> &cubeGroupOfOp) {
    SmallVector<StageGroup> cubeGroups;
    llvm::SmallPtrSet<Operation *, 32> alreadyInCube;

    for (auto &op : currentForOp.getBody()->getOperations()) {
      if (!isCubeAnchor(&op) || alreadyInCube.contains(&op))
        continue;

      StageGroup group;
      group.coreType = hivm::TCoreType::CUBE;
      addCubeBoundaryCopies(&op, group, alreadyInCube);
      addOpToGroup(group, &op);
      sortGroupOps(group);

      if (group.ops.empty())
        continue;

      std::size_t groupId = cubeGroups.size();
      for (auto *groupOp : group.ops) {
        cubeGroupOfOp[groupOp] = groupId;
        alreadyInCube.insert(groupOp);
      }
      cubeGroups.push_back(std::move(group));
    }

    return cubeGroups;
  }

  // Cube 组先由 GEMM anchor 固定下来。剩余 op 按 pipeline body 中的源码顺序归入
  // Vector 组，遇到 Cube 组就切断当前 Vector 段并插入 Cube 组。
  SmallVector<StageGroup> collectVectorGroupsByOrder(
      const SmallVector<StageGroup> &cubeGroups,
      const DenseMap<Operation *, std::size_t> &cubeGroupOfOp) {
    SmallVector<StageGroup> orderedGroups;
    SmallVector<char> emittedCube(cubeGroups.size(), 0);
    StageGroup pendingVector;
    pendingVector.coreType = hivm::TCoreType::VECTOR;

    auto flushVector = [&]() {
      if (pendingVector.ops.empty())
        return;
      orderedGroups.push_back(std::move(pendingVector));
      pendingVector = StageGroup{};
      pendingVector.coreType = hivm::TCoreType::VECTOR;
    };

    for (auto &op : currentForOp.getBody()->getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      auto cubeIt = cubeGroupOfOp.find(&op);
      if (cubeIt != cubeGroupOfOp.end()) {
        flushVector();
        std::size_t groupId = cubeIt->second;
        if (!emittedCube[groupId]) {
          orderedGroups.push_back(cubeGroups[groupId]);
          emittedCube[groupId] = 1;
        }
        continue;
      }

      if (shouldKeepOutsideScope(&op)) {
        flushVector();
        continue;
      }

      pendingVector.ops.push_back(&op);
    }

    flushVector();
    return orderedGroups;
  }

  // 在每组第一个 op 的原位置创建 scope，再按原序搬入组内 op。这样既保持 C/V stage
  // 的执行顺序，也允许 alloc/workspace 留在父循环里被多个 scope 捕获。
  void packGroups(SmallVectorImpl<StageGroup> &groups) {
    for (auto &group : groups) {
      if (group.ops.empty())
        continue;
      sortGroupOps(group);

      OpBuilder builder(group.ops.front());
      auto scope =
          builder.create<scope::ScopeOp>(builder.getUnknownLoc(), TypeRange());

      scope->setAttr(hivm::TCoreTypeAttr::name,
                     builder.getAttr<hivm::TCoreTypeAttr>(group.coreType));

      auto &scopeBody = scope.getRegion().emplaceBlock();
      for (auto *op : group.ops)
        op->moveBefore(&scopeBody, scopeBody.end());

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&scopeBody);
      builder.create<scope::ReturnOp>(builder.getUnknownLoc());

      LDBG("Packed a " << hivm::stringifyTCoreType(group.coreType)
                       << "-core scope:\n"
                       << scope);
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

  // 将任意 op 映射为 currentForOp body 的直接子 op。
  // 如果 op 在嵌套 scf.for/scf.if 内部，这里返回外层那个直接子 op；
  // 如果 op 不属于当前 pipeline loop，则返回 nullptr。
  Operation *getTopLevelOpInCurrentFor(Operation *op) {
    auto *body = currentForOp.getBody();
    Operation *cur = op;
    while (cur && cur->getBlock() != body)
      cur = cur->getParentOp();
    return cur;
  }

  // 判断一个 value 是否来自 workspace alloc；subview/reinterpret_cast 这类 view op
  // 会递归追溯到原始 source。
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

  // 纯 scalar op 不携带 tensor/memref/vector 数据，当前先留在父 loop 中，
  // 避免生成 EnableMultiBuffer 尚未处理的 scope result。
  bool isScalarOp(Operation *op) {
    if (op->getNumRegions() != 0 || op->getNumResults() == 0)
      return false;
    auto isScalar = [](Value val) {
      return !isa<TensorType, BaseMemRefType, VectorType>(val.getType());
    };
    return llvm::all_of(op->getOperands(), isScalar) &&
           llvm::all_of(op->getResults(), isScalar);
  }
};
#undef DEBUG_TYPE

} // namespace mlir::tilelangir
