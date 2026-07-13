"""Bottom-up tau-merge spatial partition (per time window) -- v2 fresh implementation.

Start from kxk pixel cells (leaves). Repeatedly merge the adjacent region pair whose
union has the smallest *residual ratio*
    rho(r u s) = E_{r u s} / (E0_{r u s} + eps_{r u s})
while rho <= tau. E is the window-summed Killing LSQ residual energy with the union's
own per-timestep optimal observer (killing2d.solve_killing); E0 is the u=0 energy
sum ||dv/dt||^2. So tau reads as: "a region may exist only while a single rigid
observer explains all but a tau-fraction of its raw temporal energy".

Design note (traceability): the first version used the *increment* criterion
    delta = [E_u - E_r - E_s] / E0_u <= tau
and was defeated by salami-slicing in the two-rotor counter-control (validate_rfc T4):
each bite of a foreign region is small relative to the union's ever-growing E0, so
chains of tau-passing merges collapsed everything to N=1. The ratio criterion bounds
the *cumulative* unexplained fraction per region, which is what the merge tolerance
is supposed to mean. RFC still collapses to N=1 (rho stays ~3e-4 for every union).

eps is a per-pixel floor (eps_rel x global mean raw energy per pixel x region pixels):
regions that are locally steady (E0 ~ 0, e.g. far-field) have rho ~ 0 and merge freely
into any neighbor, which is correct -- a steady patch is explained by any observer.

Region sufficient statistics are running sums over cells, so evaluating a candidate
merge costs one batched 3x3 solve over the window timesteps.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from killing2d import CellStats, solve_killing


@dataclass
class Region:
    cells: list            # flat cell ids
    AtA: np.ndarray        # (Tw, 3, 3) running sum
    g: np.ndarray          # (Tw, 3)
    e0: np.ndarray         # (Tw,)
    npix: float
    q: np.ndarray          # (Tw, 3) per-timestep Killing params (a, b, c)
    E: float               # window-summed residual energy at q
    E0: float              # window-summed raw energy
    version: int = 0       # bumped on every merge; stale heap entries are skipped


@dataclass
class WindowPartition:
    it0: int
    it1: int
    tau: float
    labels_cells: np.ndarray      # (nCy, nCx) int32 region index (0..N-1)
    labels_pixels: np.ndarray     # (Y, X) int32
    regions: list                 # list[Region], index == label
    merge_log: list = field(default_factory=list)   # (delta, n_regions_after)
    n_absorbed: int = 0           # tiny regions force-merged post-hoc (spec deviation)

    @property
    def n_regions(self) -> int:
        return len(self.regions)


def _make_region(stats: CellStats, cid: int, it0: int, it1: int) -> Region:
    nCy, nCx = stats.n_cells
    cy, cx = np.unravel_index(cid, (nCy, nCx))
    AtA = stats.AtA[it0:it1, cy, cx].copy()
    g = stats.g[it0:it1, cy, cx].copy()
    e0 = stats.e0[it0:it1, cy, cx].copy()
    q, E = solve_killing(AtA, g, e0)
    return Region(cells=[cid], AtA=AtA, g=g, e0=e0, npix=float(stats.npix[cy, cx]),
                  q=q, E=float(E.sum()), E0=float(e0.sum()))


def _union_eval(ra: Region, rb: Region) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                 np.ndarray, float, float]:
    AtA = ra.AtA + rb.AtA
    g = ra.g + rb.g
    e0 = ra.e0 + rb.e0
    q, E = solve_killing(AtA, g, e0)
    return AtA, g, e0, q, float(E.sum()), float(e0.sum())


def merge_partition(stats: CellStats, it0: int, it1: int, tau: float,
                    min_regions: int = 1, eps_rel: float = 1e-6,
                    absorb_min_pixels: int = 0) -> WindowPartition:
    """Greedy agglomerative tau-merge over the cell grid for window [it0, it1).

    absorb_min_pixels > 0 enables an OPTIONAL post-pass (documented spec deviation):
    every final region smaller than this is force-merged into the adjacent region
    with the lowest union residual ratio, IGNORING tau. Motivation: the uniform
    per-INR budget rule (B/M) makes stray 4-pixel regions absurdly expensive (each
    eats a full INR share); absorbing them trades a bounded rho increase for a sane
    M. Absorbed pixels are reconstructed under the host region's observer. Off (0)
    by default to preserve the pure tau semantics."""
    nCy, nCx = stats.n_cells
    n_leaf = nCy * nCx
    regions: dict[int, Region] = {cid: _make_region(stats, cid, it0, it1)
                                  for cid in range(n_leaf)}
    E0_global = sum(r.E0 for r in regions.values())
    npix_global = max(sum(r.npix for r in regions.values()), 1.0)
    eps_pix = eps_rel * E0_global / npix_global + 1e-300   # per-pixel rho floor

    # cell-grid 4-adjacency -> region adjacency sets
    adj: dict[int, set] = {cid: set() for cid in range(n_leaf)}
    for cy in range(nCy):
        for cx in range(nCx):
            cid = cy * nCx + cx
            if cx + 1 < nCx:
                adj[cid].add(cid + 1); adj[cid + 1].add(cid)
            if cy + 1 < nCy:
                adj[cid].add(cid + nCx); adj[cid + nCx].add(cid)

    def rho_of(ra: Region, rb: Region) -> tuple[float, tuple]:
        AtA, g, e0, q, E_u, E0_u = _union_eval(ra, rb)
        # hard floor guards npix==0 unions (cells fully inside the boundary-skip
        # band carry no LSQ votes): E_u = 0 there, rho = 0, merge freely
        rho = E_u / max(E0_u + eps_pix * (ra.npix + rb.npix), 1e-300)
        return rho, (AtA, g, e0, q, E_u, E0_u)

    heap: list = []
    cache: dict[tuple, tuple] = {}

    def push_pair(ia: int, ib: int):
        a, b = (ia, ib) if ia < ib else (ib, ia)
        ra, rb = regions[a], regions[b]
        rho, payload = rho_of(ra, rb)
        cache[(a, b, ra.version, rb.version)] = payload
        heapq.heappush(heap, (rho, a, b, ra.version, rb.version))

    seen_pairs = set()
    for ia, nbrs in adj.items():
        for ib in nbrs:
            key = (min(ia, ib), max(ia, ib))
            if key not in seen_pairs:
                seen_pairs.add(key)
                push_pair(*key)

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
        AtA, g, e0, q, E_u, E0_u = cache.pop((a, b, va, vb))
        ra, rb = regions.pop(a), regions.pop(b)
        rn = Region(cells=ra.cells + rb.cells, AtA=AtA, g=g, e0=e0,
                    npix=ra.npix + rb.npix, q=q, E=E_u, E0=E0_u,
                    version=0)
        rid = next_id; next_id += 1
        regions[rid] = rn
        new_nbrs = (adj[a] | adj[b]) - {a, b}
        adj[rid] = new_nbrs
        for nb in new_nbrs:
            adj[nb].discard(a); adj[nb].discard(b); adj[nb].add(rid)
        del adj[a], adj[b]
        merge_log.append((float(d), len(regions)))
        for nb in new_nbrs:
            if nb in regions:
                push_pair(rid, nb)

    # optional post-pass: absorb tiny regions into their best-fitting neighbor
    n_absorbed = 0
    if absorb_min_pixels > 0:
        while len(regions) > 1:
            small = [rid for rid, r in regions.items()
                     if len(r.cells) * stats.k * stats.k < absorb_min_pixels]
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
            rn = Region(cells=ra.cells + rb.cells, AtA=AtA, g=g, e0=e0,
                        npix=ra.npix + rb.npix, q=q, E=E_u, E0=E0_u)
            nid = next_id; next_id += 1
            regions[nid] = rn
            new_nbrs = (adj[rid] | adj[best_nb]) - {rid, best_nb}
            adj[nid] = new_nbrs
            for nb in new_nbrs:
                adj[nb].discard(rid); adj[nb].discard(best_nb); adj[nb].add(nid)
            del adj[rid], adj[best_nb]
            n_absorbed += 1

    # compact labels
    ids = sorted(regions.keys())
    labels_cells = np.full((nCy, nCx), -1, dtype=np.int32)
    out_regions = []
    for new_label, rid in enumerate(ids):
        r = regions[rid]
        cy, cx = np.unravel_index(np.asarray(r.cells, dtype=np.int64), (nCy, nCx))
        labels_cells[cy, cx] = new_label
        out_regions.append(r)
    assert (labels_cells >= 0).all(), "cell label coverage hole"

    # expand cell labels to pixel labels
    ey, ex = stats.cell_y0, stats.cell_x0
    reps_y = np.diff(ey); reps_x = np.diff(ex)
    labels_pixels = np.repeat(np.repeat(labels_cells, reps_y, axis=0), reps_x, axis=1)
    assert labels_pixels.shape == (stats.Y, stats.X)
    assert (labels_pixels >= 0).all(), "pixel label coverage hole"

    return WindowPartition(it0=it0, it1=it1, tau=tau, labels_cells=labels_cells,
                           labels_pixels=labels_pixels, regions=out_regions,
                           merge_log=merge_log, n_absorbed=n_absorbed)


def split_windows(T: int, n_windows: int, allow_full: bool = False) -> list[tuple[int, int]]:
    """Contiguous near-equal windows over timestep indices. Enforces the spec rule
    'window length <= half the time extent' (i.e. n_windows >= 2) unless T < 4.
    allow_full=True bypasses the rule for DIAGNOSTIC runs only (attribution of the
    window-split cost); such runs must be labeled as spec-violating."""
    if T >= 4 and n_windows < 2 and not allow_full:
        raise ValueError("time window must be <= half of [tmin, tmax]: need n_windows >= 2")
    edges = np.linspace(0, T, n_windows + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_windows)]
