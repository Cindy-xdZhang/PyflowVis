"""
FLowUtils/referenceFrameDecompose.py

把一个 2D 非定常流场**空间划分**成若干区域，每个区域可用一个（时变的）刚体 Killing
observer 解释——即"分片最优参考系"。自底向上区域合并，产出一棵 **merge tree
(dendrogram)**，可在任意粒度 cut。用于诊断"是否存在多个流向（单一全局 observer 不够）"
以及后续下游任务（分片 observer / 分片 observed field / 压缩等）。

方法（详见 docs/referenceframeOptimization.md）：
  - 每个像素、每帧构造 3-DOF Killing 最小二乘 A·q=b（q=(a,b,c)，u=(a,b)+c(-y,x)，b=-∂v/∂t）。
  - 每个 leaf(kxk) 累加**充分统计量** S=ΣAᵀA, r=ΣAᵀb, e=Σbᵀb（逐帧）。合并区域 = 相加统计量。
  - 区域残差（= observed time derivative 能量）E = Σ_t (e_t - r_tᵀ S_t⁻¹ r_t)，闭式、可累加。
  - Region Adjacency Graph + 优先队列，每步合并"代价 Δ=E_AB-E_A-E_B 最小"的相邻区域（Ward 式 linkage），
    一直合并到 1 个区域，记录 dendrogram。之后按需 cut(n_regions / cost_threshold)。

设计决策（已与用户对齐）：完整 dendrogram 事后 cut；observer 3 DOF；静态空间划分（一棵树跨帧累加）。

关键实现点：J=∇v 用**全局有限差分**（leaf 太小算不准）；充分统计量逐帧累加到 leaf（省内存）；
残差批量解（T 个 3x3）；优先队列 lazy deletion。
"""
import heapq
from dataclasses import dataclass, field as _dc_field
from typing import Dict, Optional, Tuple

import numpy as np

from .VectorField2d import UnsteadyVectorField2D
from .ReferenceFrameOptimization import _as_numpy, _dvdt, _coords_2d, _make_like_2d


# ═══════════════════════════════════════════════════════════════════════════
# Leaf sufficient statistics
# ═══════════════════════════════════════════════════════════════════════════
def _leaf_sufficient_stats(field: UnsteadyVectorField2D, k: int):
    """逐帧累加每个 kxk leaf 的充分统计量。

    Returns:
        leaf_S (L,T,3,3), leaf_r (L,T,3), leaf_e (L,T), leaf_n (L,),
        nly, nlx, leaf_of_pixel (Y,X int)。
    """
    data = _as_numpy(field)                              # (T,Y,X,2) float64
    T, Y, X, _ = data.shape
    dx, dy = float(field.gridInterval[0]), float(field.gridInterval[1])
    dt = float(field.timeInterval)
    Xg, Yg = _coords_2d(field)                           # (Y,X) physical, time-independent
    Xp = np.stack([-Yg, Xg], axis=-1)                    # (Y,X,2)  position perp
    vt_all = _dvdt(data, dt)                             # (T,Y,X,2)

    ys = np.arange(0, Y, k); xs = np.arange(0, X, k)     # block starts (reduceat handles remainder)
    nly, nlx = len(ys), len(xs)
    L = nly * nlx

    leaf_S = np.zeros((L, T, 3, 3), np.float64)
    leaf_r = np.zeros((L, T, 3), np.float64)
    leaf_e = np.zeros((L, T), np.float64)

    for t in range(T):
        sl = data[t]                                     # (Y,X,2)
        dv_dx = np.gradient(sl, dx, axis=1)              # (Y,X,2) = ∂v/∂x
        dv_dy = np.gradient(sl, dy, axis=0)              # ∂v/∂y
        J = np.stack([dv_dx, dv_dy], axis=-1)            # (Y,X,2,2), J[...,i,j]=∂v_i/∂x_j
        Vp = np.stack([-sl[:, :, 1], sl[:, :, 0]], -1)   # (-v_y, v_x)
        JXp = np.einsum("yxik,yxk->yxi", J, Xp)          # J·Xp
        A = np.empty((Y, X, 2, 3), np.float64)           # [∂v/∂x | ∂v/∂y | J·Xp − Vp]
        A[:, :, :, 0] = dv_dx
        A[:, :, :, 1] = dv_dy
        A[:, :, :, 2] = JXp - Vp
        b = -vt_all[t]                                   # (Y,X,2) = −∂v/∂t

        AtA = np.einsum("yxki,yxkj->yxij", A, A)         # (Y,X,3,3)
        Atb = np.einsum("yxki,yxk->yxi", A, b)           # (Y,X,3)
        btb = np.einsum("yxk,yxk->yx", b, b)             # (Y,X)

        # sum over kxk blocks (reduceat handles non-divisible sizes)
        S_t = np.add.reduceat(np.add.reduceat(AtA, ys, 0), xs, 1).reshape(L, 3, 3)
        r_t = np.add.reduceat(np.add.reduceat(Atb, ys, 0), xs, 1).reshape(L, 3)
        e_t = np.add.reduceat(np.add.reduceat(btb, ys, 0), xs, 1).reshape(L)
        leaf_S[:, t] = S_t; leaf_r[:, t] = r_t; leaf_e[:, t] = e_t

    ones = np.ones((Y, X), np.float64)
    leaf_n = np.add.reduceat(np.add.reduceat(ones, ys, 0), xs, 1).reshape(L)

    iy = (np.arange(Y) // k)[:, None]
    ix = (np.arange(X) // k)[None, :]
    leaf_of_pixel = (iy * nlx + ix).astype(np.int32)     # (Y,X)
    return leaf_S, leaf_r, leaf_e, leaf_n, nly, nlx, leaf_of_pixel


def _residual(S: np.ndarray, r: np.ndarray, e: np.ndarray, eps=1e-8, floor=1e-12) -> float:
    """observed-td 残差能量 E = Σ_t (e_t − r_tᵀ S_t⁻¹ r_t)，逐帧批量、带正则稳健化。
    S (T,3,3), r (T,3), e (T,)."""
    tr = np.einsum("tii->t", S) / 3.0
    reg = S + (eps * tr + floor)[:, None, None] * np.eye(3)
    q = np.linalg.solve(reg, r[..., None])[..., 0]        # (T,3)
    res_t = e - np.einsum("ti,ti->t", r, q)               # (T,)
    return float(np.maximum(res_t, 0.0).sum())


def _observer(S: np.ndarray, r: np.ndarray, eps=1e-8, floor=1e-12) -> np.ndarray:
    """区域的逐帧 Killing 系数 (T,3)=(a,b,c)。"""
    tr = np.einsum("tii->t", S) / 3.0
    reg = S + (eps * tr + floor)[:, None, None] * np.eye(3)
    return np.linalg.solve(reg, r[..., None])[..., 0]


# ═══════════════════════════════════════════════════════════════════════════
# Result: merge tree + cut / observers / diagnostics
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ReferenceFrameDecomposition:
    """分片最优参考系的 merge tree 结果。用 cut(...) 取任意粒度的分区。"""
    field: UnsteadyVectorField2D
    k: int
    # leaf grid
    nly: int
    nlx: int
    leaf_of_pixel: np.ndarray            # (Y,X) int -> leaf id
    # leaf sufficient stats (kept for observer/residual at any cut)
    leaf_S: np.ndarray                   # (L,T,3,3)
    leaf_r: np.ndarray                   # (L,T,3)
    leaf_e: np.ndarray                   # (L,T)
    leaf_n: np.ndarray                   # (L,)
    # dendrogram: each merge (child_a, child_b, cost, residual, size); node id = L + i
    linkage: np.ndarray                  # (n_merges, 5)
    leaf_rep: np.ndarray                 # (n_nodes,) a representative leaf per node
    diag: dict = _dc_field(default_factory=dict)

    @property
    def n_leaves(self) -> int:
        return self.leaf_S.shape[0]

    # ---- cut the dendrogram ----
    def cut(self, n_regions: Optional[int] = None, cost_threshold: Optional[float] = None) -> np.ndarray:
        """返回像素级 labels (Y,X)。给 n_regions 取固定区域数；给 cost_threshold 合并代价超阈值即停。"""
        L = self.n_leaves
        if n_regions is not None:
            n_merges_keep = max(0, L - int(n_regions))
        elif cost_threshold is not None:
            costs = self.linkage[:, 2]
            # prefix: keep merges up to the FIRST one exceeding the threshold ("首次明显上升即停").
            # (count(costs<=thr) would wrongly re-include a cheap merge occurring AFTER an expensive
            #  one, since Ward-style costs are only near-monotone, not strictly monotone.)
            over = np.where(costs > cost_threshold)[0]
            n_merges_keep = int(over[0]) if over.size else int(self.linkage.shape[0])
        else:
            raise ValueError("give n_regions or cost_threshold")
        parent = np.arange(L)
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for i in range(min(n_merges_keep, self.linkage.shape[0])):
            a, b = int(self.linkage[i, 0]), int(self.linkage[i, 1])
            ra, rb = find(self.leaf_rep[a]), find(self.leaf_rep[b])
            if ra != rb:
                parent[rb] = ra
        roots = np.array([find(i) for i in range(L)])
        _, labels_leaf = np.unique(roots, return_inverse=True)   # 0..n-1 contiguous
        return labels_leaf.astype(np.int32)[self.leaf_of_pixel]

    # ---- adaptive cut: auto-determine #regions (no manual cut) ----
    def cut_adaptive(self, rel_threshold: float = 0.01):
        """自适应决定最终区域数（无需人为指定 cut）。

        自底向上合并：两相邻区域，各自算最优 killing observer 与合并后的 observer；若"合并后 observed
        time derivative 能量不明显上升"就合并，首次明显上升即停。合并代价
        Δ = E_AB − E_A − E_B ≥ 0（强迫两区共用一个 killing observer 的残差增量；Δ≈0 表示能共用）。
        "明显上升"= Δ > rel_threshold · raw_td_energy（相对整场原始非定常能量，scale-invariant）。

        Args:
            rel_threshold: 合并容忍度（Δ 相对整场 ∂v/∂t 能量的比例）。越小 → 区域越多（越严格）。
        Returns:
            (labels (Y,X) int, n_regions)。
        """
        raw = max(float(self.diag.get("raw_td_energy", 0.0)), 1e-30)
        labels = self.cut(cost_threshold=rel_threshold * raw)   # prefix-based stop (first Δ>thr)
        n_regions = len(np.unique(labels))
        return labels, n_regions

    def merge_cost_curve(self):
        """返回每步合并代价 Δ（升序）与对应的剩余区域数，用于观察肘部 / 选 rel_threshold。
        Returns: (n_regions_after[], cost[], cost/raw_td_energy[])。"""
        costs = self.linkage[:, 2] if self.linkage.size else np.zeros(0)
        n_after = self.n_leaves - 1 - np.arange(len(costs))   # regions remaining after each merge
        raw = max(float(self.diag.get("raw_td_energy", 0.0)), 1e-30)
        return n_after, costs, costs / raw

    def _leaf_labels(self, labels: np.ndarray) -> np.ndarray:
        """把像素 labels (Y,X) 归约到每个 leaf 一个 label (L,)（取 leaf 左上角像素，label 在 leaf 内恒定）。"""
        L = self.n_leaves
        rows = (np.arange(L) // self.nlx) * self.k
        cols = (np.arange(L) % self.nlx) * self.k
        return labels[rows, cols]

    # ---- per-region observers / residuals for a given labeling ----
    def region_observers(self, labels: np.ndarray) -> Dict[int, np.ndarray]:
        """每个区域的逐帧 Killing 系数 (T,3)=(a,b,c)。labels: (Y,X)。"""
        leaf_label = self._leaf_labels(labels)
        out = {}
        for rid in np.unique(leaf_label):
            m = leaf_label == rid
            out[int(rid)] = _observer(self.leaf_S[m].sum(0), self.leaf_r[m].sum(0))
        return out

    def region_residuals(self, labels: np.ndarray) -> Dict[int, float]:
        """每个区域的 observed-td 残差能量。labels: (Y,X)。"""
        leaf_label = self._leaf_labels(labels)
        out = {}
        for rid in np.unique(leaf_label):
            m = leaf_label == rid
            out[int(rid)] = _residual(self.leaf_S[m].sum(0), self.leaf_r[m].sum(0), self.leaf_e[m].sum(0))
        return out

    # ---- observer field for a labeling (piecewise per-region rigid observer) ----
    def observer_field(self, labels: np.ndarray) -> UnsteadyVectorField2D:
        """按分区把每区域的 Killing observer 铺成整场 u(x,t)（分片刚体）。"""
        f = self.field
        data = _as_numpy(f); T, Y, X, _ = data.shape
        Xg, Yg = _coords_2d(f)
        u = np.zeros((T, Y, X, 2), np.float64)
        u0, u1 = u[..., 0], u[..., 1]                        # (T,Y,X) views into u
        for rid, coeff in self.region_observers(labels).items():   # coeff (T,3)
            mask = labels == rid                             # (Y,X)
            a = coeff[:, 0][:, None, None]; b = coeff[:, 1][:, None, None]; c = coeff[:, 2][:, None, None]
            ux = a - c * Yg[None]                            # (T,Y,X)
            uy = b + c * Xg[None]
            u0[:, mask] = ux[:, mask]
            u1[:, mask] = uy[:, mask]
        return _make_like_2d(f, u)

    # ---- residual vs #regions curve (to choose a cut) ----
    def residual_curve(self, max_regions: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (n_regions[], total_observed_residual[])，从 1 到 max_regions。用于选 cut 点（肘部）。"""
        L = self.n_leaves
        ns = list(range(1, min(max_regions, L) + 1))
        res = []
        for n in ns:
            labels = self.cut(n_regions=n)
            res.append(sum(self.region_residuals(labels).values()))
        return np.array(ns), np.array(res)


# ═══════════════════════════════════════════════════════════════════════════
# Build the decomposition
# ═══════════════════════════════════════════════════════════════════════════
def decompose_reference_frame_2d(field: UnsteadyVectorField2D, k: int = 2,
                                 verbose: bool = False) -> ReferenceFrameDecomposition:
    """自底向上区域合并，构建分片最优参考系的 merge tree。

    Args:
        field: 非定常 2D 流场。
        k:     最小 leaf 单元边长（kxk 像素，默认 2）。
    Returns:
        ReferenceFrameDecomposition（用 .cut(n_regions=...) 取分区，.diag 看诊断）。
    """
    leaf_S, leaf_r, leaf_e, leaf_n, nly, nlx, leaf_of_pixel = _leaf_sufficient_stats(field, k)
    L = leaf_S.shape[0]

    # per-region running stats + residual (start = leaves)
    S = {i: leaf_S[i].copy() for i in range(L)}
    r = {i: leaf_r[i].copy() for i in range(L)}
    e = {i: leaf_e[i].copy() for i in range(L)}
    E = {i: _residual(S[i], r[i], e[i]) for i in range(L)}
    alive = np.ones(L * 2, bool)  # will grow; ids 0..L-1 leaves, L.. internal

    # Region Adjacency Graph (4-neighbour on leaf grid)
    adj: Dict[int, set] = {i: set() for i in range(L)}
    def lid(iy, ix): return iy * nlx + ix
    for iy in range(nly):
        for ix in range(nlx):
            i = lid(iy, ix)
            if ix + 1 < nlx: adj[i].add(lid(iy, ix + 1)); adj[lid(iy, ix + 1)].add(i)
            if iy + 1 < nly: adj[i].add(lid(iy + 1, ix)); adj[lid(iy + 1, ix)].add(i)

    def merge_cost(a, bb):
        Emab = _residual(S[a] + S[bb], r[a] + r[bb], e[a] + e[bb])
        return Emab - E[a] - E[bb], Emab

    heap = []
    cnt = 0
    for a in adj:
        for bb in adj[a]:
            if a < bb:
                cost, _ = merge_cost(a, bb)
                heapq.heappush(heap, (cost, cnt, a, bb)); cnt += 1

    # dendrogram storage
    leaf_rep = list(range(L)) + [0] * (L - 1)   # representative leaf per node
    linkage = []                                # (a,b,cost,residual,size)
    size = {i: 1 for i in range(L)}
    next_id = L

    while heap:
        cost, _, a, bb = heapq.heappop(heap)
        if not alive[a] or not alive[bb] or bb not in adj.get(a, ()):
            continue                            # stale edge (lazy deletion)
        # create merged region c
        c = next_id; next_id += 1
        if c >= alive.shape[0]:
            alive = np.concatenate([alive, np.ones(alive.shape[0], bool)])
        S[c] = S[a] + S[bb]; r[c] = r[a] + r[bb]; e[c] = e[a] + e[bb]
        E[c] = E[a] + E[bb] + cost
        size[c] = size[a] + size[bb]
        leaf_rep[c] = leaf_rep[a]
        linkage.append((a, bb, cost, E[c], size[c]))
        # rewire adjacency
        neigh = (adj[a] | adj[bb]) - {a, bb}
        adj[c] = set()
        for nb in neigh:
            if not alive[nb]:
                continue
            adj[nb].discard(a); adj[nb].discard(bb); adj[nb].add(c); adj[c].add(nb)
            newcost, _ = merge_cost(c, nb)
            heapq.heappush(heap, (newcost, cnt, min(c, nb), max(c, nb))); cnt += 1
        alive[a] = alive[bb] = False
        adj.pop(a, None); adj.pop(bb, None)
        # free stats of consumed regions
        for x in (a, bb):
            S.pop(x, None); r.pop(x, None); e.pop(x, None)
        if verbose and len(linkage) % max(1, L // 10) == 0:
            print(f"  merged {len(linkage)}/{L-1}  regions_left={L - len(linkage)}  last_cost={cost:.4e}")

    linkage = np.array(linkage, np.float64) if linkage else np.zeros((0, 5))
    leaf_rep = np.array(leaf_rep, np.int64)

    # ---- diagnostics: multi-flow vs intrinsic unsteadiness ----
    raw_energy = float(leaf_e.sum())                         # ‖∂v/∂t‖²  (u=0)
    global_res = _residual(leaf_S.sum(0), leaf_r.sum(0), leaf_e.sum(0))   # single global observer
    finest_res = float(sum(_residual(leaf_S[i], leaf_r[i], leaf_e[i]) for i in range(L)))
    diag = dict(
        raw_td_energy=raw_energy,
        global_observer_residual=global_res,
        finest_observer_residual=finest_res,
        global_residual_ratio=global_res / max(raw_energy, 1e-30),   # 1 global observer 剩多少非定常
        finest_residual_ratio=finest_res / max(raw_energy, 1e-30),   # 最细分区仍剩多少（大 => 本征非定常）
        # 分区相对原始能量能多消除的非定常比例（主判据；对小残差稳健，不像比值被小分母放大）
        decomposition_benefit=(global_res - finest_res) / max(raw_energy, 1e-30),
        decomposition_gain=global_res / max(finest_res, 1e-30),      # 参考：比值（小残差时会虚高）
        n_leaves=L, leaf_grid=(nly, nlx),
    )
    diag["interpretation"] = _interpret(diag)
    if verbose:
        print("[decompose] diagnostics:", {k: (round(v, 4) if isinstance(v, float) else v)
                                            for k, v in diag.items() if k != "interpretation"})
        print("  ", diag["interpretation"])

    return ReferenceFrameDecomposition(
        field=field, k=k, nly=nly, nlx=nlx, leaf_of_pixel=leaf_of_pixel,
        leaf_S=leaf_S, leaf_r=leaf_r, leaf_e=leaf_e, leaf_n=leaf_n,
        linkage=linkage, leaf_rep=leaf_rep, diag=diag)


def _interpret(d: dict) -> str:
    gr, fr, ben = d["global_residual_ratio"], d["finest_residual_ratio"], d["decomposition_benefit"]
    if gr < 0.05:
        return "单一全局 observer 已足够（几乎全局刚体可解释），无需多区域划分。"
    if fr > 0.5:
        return ("最细分区残差仍大 => 场偏**本征非定常**（涡脱落/湍流等），任何刚体分区都压不平，"
                "划分收益有限。")
    if ben > 0.2:
        return ("存在**多个流向**：分区相对原始非定常能多消除 %.0f%%，空间划分显著有益（cut 出多个区域）。" % (ben * 100))
    return "介于两者之间：分区有一定收益，用 residual_curve() 找肘部选 cut。"


__all__ = ["ReferenceFrameDecomposition", "decompose_reference_frame_2d"]
