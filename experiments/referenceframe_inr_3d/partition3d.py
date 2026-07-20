"""Bottom-up tau-merge spatial partition (per time window) -- 3D lift.

Same algorithm and ratio criterion as experiments/referenceframe_inr_v2/
partition.py (see its docstring for the salami-slicing design note); the only
3D changes are the cell grid (kxkxk voxels, flat ids over (nCz, nCy, nCx)),
6-adjacency, and the voxel-label expansion.  Region statistics are running sums
over cells, so a candidate merge costs one batched 6x6 solve over the window.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from killing3d import CellStats3D, solve_killing_3d


@dataclass
class Region3D:
    cells: list            # flat cell ids
    AtA: np.ndarray        # (Tw, 6, 6) running sum
    g: np.ndarray          # (Tw, 6)
    e0: np.ndarray         # (Tw,)
    npix: float
    q: np.ndarray          # (Tw, 6) per-timestep Killing params (t_vec, w)
    E: float
    E0: float
    version: int = 0


@dataclass
class WindowPartition3D:
    it0: int
    it1: int
    tau: float
    labels_cells: np.ndarray      # (nCz, nCy, nCx) int32 region index
    labels_pixels: np.ndarray     # (Z, Y, X) int32
    regions: list
    merge_log: list = field(default_factory=list)
    n_absorbed: int = 0

    @property
    def n_regions(self) -> int:
        return len(self.regions)


def _make_region(stats: CellStats3D, cid: int, it0: int, it1: int) -> Region3D:
    nC = stats.n_cells
    cz, cy, cx = np.unravel_index(cid, nC)
    AtA = stats.AtA[it0:it1, cz, cy, cx].copy()
    g = stats.g[it0:it1, cz, cy, cx].copy()
    e0 = stats.e0[it0:it1, cz, cy, cx].copy()
    q, E = solve_killing_3d(AtA, g, e0)
    return Region3D(cells=[cid], AtA=AtA, g=g, e0=e0,
                    npix=float(stats.npix[cz, cy, cx]),
                    q=q, E=float(E.sum()), E0=float(e0.sum()))


def _union_eval(ra: Region3D, rb: Region3D):
    AtA = ra.AtA + rb.AtA
    g = ra.g + rb.g
    e0 = ra.e0 + rb.e0
    q, E = solve_killing_3d(AtA, g, e0)
    return AtA, g, e0, q, float(E.sum()), float(e0.sum())


def merge_partition_3d(stats: CellStats3D, it0: int, it1: int, tau: float,
                       min_regions: int = 1, eps_rel: float = 1e-6,
                       absorb_min_pixels: int = 0) -> WindowPartition3D:
    """Greedy agglomerative tau-merge over the 3D cell grid for [it0, it1).
    absorb_min_pixels: same optional post-pass as 2D (force-merge regions smaller
    than this many voxels into their best-fitting neighbor, ignoring tau)."""
    nCz, nCy, nCx = stats.n_cells
    n_leaf = nCz * nCy * nCx
    regions: dict[int, Region3D] = {cid: _make_region(stats, cid, it0, it1)
                                    for cid in range(n_leaf)}
    E0_global = sum(r.E0 for r in regions.values())
    npix_global = max(sum(r.npix for r in regions.values()), 1.0)
    eps_pix = eps_rel * E0_global / npix_global + 1e-300

    # cell-grid 6-adjacency -> region adjacency sets
    adj: dict[int, set] = {cid: set() for cid in range(n_leaf)}
    for cz in range(nCz):
        for cy in range(nCy):
            for cx in range(nCx):
                cid = (cz * nCy + cy) * nCx + cx
                if cx + 1 < nCx:
                    adj[cid].add(cid + 1); adj[cid + 1].add(cid)
                if cy + 1 < nCy:
                    adj[cid].add(cid + nCx); adj[cid + nCx].add(cid)
                if cz + 1 < nCz:
                    adj[cid].add(cid + nCy * nCx); adj[cid + nCy * nCx].add(cid)

    def rho_of(ra: Region3D, rb: Region3D):
        AtA, g, e0, q, E_u, E0_u = _union_eval(ra, rb)
        rho = E_u / max(E0_u + eps_pix * (ra.npix + rb.npix), 1e-300)
        return rho, (AtA, g, e0, q, E_u, E0_u)

    # Unlike the 2D module there is NO payload cache: 3D payloads are ~6 KB each
    # and stale heap entries would pin 10^5 of them (~GBs).  The winning pair's
    # union is re-evaluated once after the pop instead (one batched 6x6 solve).
    #
    # Perf: candidate evaluation is BATCHED.  Per-pair np.linalg.solve on a
    # (Tw, 6, 6) stack costs ~ms in numpy dispatch overhead alone; over the
    # O(merges x degree) pair evaluations of a 10^4-cell grid that summed to
    # hours (first run: T4 took 7 h).  Stacking K pairs into one
    # (K, Tw, 6, 6) solve amortizes it ~30x.
    heap: list = []

    def push_pairs(pairs):
        pairs = [(a, b) if a < b else (b, a) for (a, b) in pairs]
        if not pairs:
            return
        K = len(pairs)
        Tw = it1 - it0
        AtA = np.empty((K, Tw, 6, 6)); gg = np.empty((K, Tw, 6))
        ee = np.empty((K, Tw)); npx = np.empty(K)
        for i, (a, b) in enumerate(pairs):
            ra, rb = regions[a], regions[b]
            AtA[i] = ra.AtA + rb.AtA
            gg[i] = ra.g + rb.g
            ee[i] = ra.e0 + rb.e0
            npx[i] = ra.npix + rb.npix
        _, E = solve_killing_3d(AtA, gg, ee)
        rho = E.sum(axis=1) / np.maximum(ee.sum(axis=1) + eps_pix * npx, 1e-300)
        for i, (a, b) in enumerate(pairs):
            ra, rb = regions[a], regions[b]
            heapq.heappush(heap, (float(rho[i]), a, b, ra.version, rb.version))

    seen_pairs = set()
    for ia, nbrs in adj.items():
        for ib in nbrs:
            key = (min(ia, ib), max(ia, ib))
            if key not in seen_pairs:
                seen_pairs.add(key)
    seed_pairs = sorted(seen_pairs)
    for s in range(0, len(seed_pairs), 4096):     # ~10 MB per 4096-pair chunk
        push_pairs(seed_pairs[s:s + 4096])

    merge_log = []
    next_id = n_leaf
    while heap and len(regions) > min_regions:
        d, a, b, va, vb = heapq.heappop(heap)
        if a not in regions or b not in regions:
            continue
        if regions[a].version != va or regions[b].version != vb:
            continue
        if d > tau:
            break
        _, (AtA, g, e0, q, E_u, E0_u) = rho_of(regions[a], regions[b])
        ra, rb = regions.pop(a), regions.pop(b)
        rn = Region3D(cells=ra.cells + rb.cells, AtA=AtA, g=g, e0=e0,
                      npix=ra.npix + rb.npix, q=q, E=E_u, E0=E0_u, version=0)
        rid = next_id; next_id += 1
        regions[rid] = rn
        new_nbrs = (adj[a] | adj[b]) - {a, b}
        adj[rid] = new_nbrs
        for nb in new_nbrs:
            adj[nb].discard(a); adj[nb].discard(b); adj[nb].add(rid)
        del adj[a], adj[b]
        merge_log.append((float(d), len(regions)))
        push_pairs([(rid, nb) for nb in new_nbrs if nb in regions])

    n_absorbed = 0
    if absorb_min_pixels > 0:
        while len(regions) > 1:
            small = [rid for rid, r in regions.items()
                     if len(r.cells) * stats.k ** 3 < absorb_min_pixels]
            if not small:
                break
            rid = min(small, key=lambda i: len(regions[i].cells))
            nbrs = [nb for nb in adj[rid] if nb in regions]
            if not nbrs:
                break
            best_nb, best_rho, best_payload = None, None, None
            for nb in nbrs:
                rho, payload = rho_of(regions[rid], regions[nb])
                if best_rho is None or rho < best_rho:
                    best_nb, best_rho, best_payload = nb, rho, payload
            AtA, g, e0, q, E_u, E0_u = best_payload
            ra, rb = regions.pop(rid), regions.pop(best_nb)
            rn = Region3D(cells=ra.cells + rb.cells, AtA=AtA, g=g, e0=e0,
                          npix=ra.npix + rb.npix, q=q, E=E_u, E0=E0_u)
            nid = next_id; next_id += 1
            regions[nid] = rn
            new_nbrs = (adj[rid] | adj[best_nb]) - {rid, best_nb}
            adj[nid] = new_nbrs
            for nb in new_nbrs:
                adj[nb].discard(rid); adj[nb].discard(best_nb); adj[nb].add(nid)
            del adj[rid], adj[best_nb]
            n_absorbed += 1

    ids = sorted(regions.keys())
    labels_cells = np.full((nCz, nCy, nCx), -1, dtype=np.int32)
    out_regions = []
    for new_label, rid in enumerate(ids):
        r = regions[rid]
        cz, cy, cx = np.unravel_index(np.asarray(r.cells, dtype=np.int64),
                                      (nCz, nCy, nCx))
        labels_cells[cz, cy, cx] = new_label
        out_regions.append(r)
    assert (labels_cells >= 0).all(), "cell label coverage hole"

    reps_z = np.diff(stats.cell_z0)
    reps_y = np.diff(stats.cell_y0)
    reps_x = np.diff(stats.cell_x0)
    labels_pixels = np.repeat(labels_cells, reps_z, axis=0)
    labels_pixels = np.repeat(labels_pixels, reps_y, axis=1)
    labels_pixels = np.repeat(labels_pixels, reps_x, axis=2)
    assert labels_pixels.shape == (stats.Z, stats.Y, stats.X)
    assert (labels_pixels >= 0).all(), "voxel label coverage hole"

    return WindowPartition3D(it0=it0, it1=it1, tau=tau, labels_cells=labels_cells,
                             labels_pixels=labels_pixels, regions=out_regions,
                             merge_log=merge_log, n_absorbed=n_absorbed)


def single_region_partition(stats: CellStats3D, it0: int, it1: int
                            ) -> WindowPartition3D:
    """M=1 fast path: the whole domain as one region, no merge walk.  Same
    result as tau >= 1 (rho < 1 always, so everything merges) but O(1) instead
    of a ~20-minute 2000-cell agglomeration on deltaWing-sized grids.  Use for
    the single-window single-region structure (the 2D winning configuration)."""
    nCz, nCy, nCx = stats.n_cells
    AtA = stats.AtA[it0:it1].sum(axis=(1, 2, 3))
    g = stats.g[it0:it1].sum(axis=(1, 2, 3))
    e0 = stats.e0[it0:it1].sum(axis=(1, 2, 3))
    q, E = solve_killing_3d(AtA, g, e0)
    reg = Region3D(cells=list(range(nCz * nCy * nCx)), AtA=AtA, g=g, e0=e0,
                   npix=float(stats.npix.sum()), q=q, E=float(E.sum()),
                   E0=float(e0.sum()))
    labels_cells = np.zeros((nCz, nCy, nCx), dtype=np.int32)
    labels_pixels = np.zeros((stats.Z, stats.Y, stats.X), dtype=np.int32)
    return WindowPartition3D(it0=it0, it1=it1, tau=-1.0,
                             labels_cells=labels_cells,
                             labels_pixels=labels_pixels, regions=[reg])


def split_windows(T: int, n_windows: int, allow_full: bool = False
                  ) -> list[tuple[int, int]]:
    """Contiguous near-equal windows; n_windows=1 requires allow_full (same rule
    as v2 after the 2026-07-17 spec change: single window is a legitimate arm)."""
    if T >= 4 and n_windows < 2 and not allow_full:
        raise ValueError("need n_windows >= 2 unless allow_full_window is set")
    edges = np.linspace(0, T, n_windows + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_windows)]
