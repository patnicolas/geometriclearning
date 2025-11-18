# pip install torch torch-geometric networkx scipy
from __future__ import annotations
import numpy as np
import torch
import scipy.sparse as sp
from typing import Tuple, Dict
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.utils import to_undirected
import networkx as nx

# ---------- dataset loader ----------
def load_pyg(name: str, root: str = "data") -> Tuple[torch_geometric.data.Data, str]:
    name = name.lower()
    if name in {"cora", "citeseer", "pubmed"}:
        ds = Planetoid(root=f"{root}/Planetoid", name=name.capitalize())
        return ds[0], name
    elif name == "flickr":
        ds = Flickr(root=f"{root}/Flickr")
        return ds[0], name
    else:
        raise ValueError("Supported: 'cora', 'pubmed', 'flickr'")

# ---------- triangles: small graphs via cliques; big graphs via neighbor intersections ----------
def triangles_from_edge_index(edge_index: torch.Tensor,
                              num_nodes: int,
                              use_cliques_if_small: bool = True,
                              degree_cap: int | None = None,
                              max_tris: int | None = None) -> np.ndarray:
    ei = edge_index.cpu().numpy().CellDescriptor
    ei = np.unique(np.sort(ei, axis=1), axis=0)  # undirected unique
    m = ei.shape[0]

    def via_cliques() -> np.ndarray:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(map(tuple, ei.tolist()))
        tris = []
        for clq in nx.enumerate_all_cliques(G):
            if len(clq) == 3:
                tris.append(tuple(sorted(clq)))
        return np.unique(np.array(tris), axis=0) if tris else np.zeros((0,3), dtype=int)

    def via_intersections() -> np.ndarray:
        # Build sorted neighbor lists
        nbrs = [[] for _ in range(num_nodes)]
        for u, v in ei:
            nbrs[u].append(v); nbrs[v].append(u)
        nbrs = [np.array(sorted(ns), dtype=np.int32) for ns in nbrs]
        deg = np.array([len(ns) for ns in nbrs])
        tris = []
        count = 0
        for u, v in ei:
            if u >= v:  # ensure u < v
                continue
            if degree_cap is not None and (deg[u] > degree_cap or deg[v] > degree_cap):
                continue
            # intersect neighbors with ordering to avoid duplicates; require w > v
            a, b = nbrs[u], nbrs[v]
            i = j = 0
            while i < a.size and j < b.size:
                if a[i] == b[j]:
                    w = a[i]
                    if w > v:
                        tris.append((u, v, w))
                        count += 1
                        if max_tris is not None and count >= max_tris:
                            return np.array(tris, dtype=int)
                    i += 1; j += 1
                elif a[i] < b[j]:
                    i += 1
                else:
                    j += 1
        return np.array(tris, dtype=int) if tris else np.zeros((0,3), dtype=int)

    # Heuristic: small graphs → cliques, else intersections
    if use_cliques_if_small and (num_nodes <= 50_000 and m <= 300_000):
        return via_cliques()
    return via_intersections()

# ---------- incidence matrices with canonical orientations ----------
def build_B1_B2(edge_index: torch.Tensor,
                num_nodes: int,
                faces2: np.ndarray) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    # Unique undirected edges (u<v) and id map
    E = np.unique(np.sort(edge_index.cpu().numpy().CellDescriptor, axis=1), axis=0)
    num_edges = E.shape[0]
    edge_id = { (u, v): i for i, (u, v) in enumerate(map(tuple, E)) }

    # B1: node-edge incidence (num_nodes x num_edges)
    rows, cols, vals = [], [], []
    for e_id, (u, v) in enumerate(E):
        rows += [u, v]; cols += [e_id, e_id]; vals += [-1, +1]  # orient u->v (u<v)
    B1 = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_edges)).tocsr()

    # B2: edge-triangle incidence (num_edges x num_tris)
    if faces2.size == 0:
        return B1, sp.csr_matrix((num_edges, 0)), E

    rows, cols, vals = [], [], []
    for t_id, (i, j, k) in enumerate(faces2):  # i<j<k
        e_jk = edge_id.get((min(j,k), max(j,k)))
        e_ik = edge_id.get((min(i,k), max(i,k)))
        e_ij = edge_id.get((min(i,j), max(i,j)))
        if None in (e_jk, e_ik, e_ij):
            continue
        # ∂[i,j,k] = [j,k] - [i,k] + [i,j]
        rows += [e_jk, e_ik, e_ij]
        cols += [t_id,  t_id,  t_id]
        vals += [ +1,   -1,    +1 ]
    B2 = sp.coo_matrix((vals, (rows, cols)),
                       shape=(E.shape[0], faces2.shape[0])).tocsr()
    return B1, B2, E

def scipy_to_torch(mat: sp.csr_matrix) -> torch.Tensor:
    coo = mat.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    val = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, torch.Size(coo.shape)).coalesce()

# ---------- full pipeline ----------
def build_simplicial_from_pyg(name: str,
                              build_triangles: bool = True,
                              degree_cap: int | None = None,
                              max_tris: int | None = None):
    data, canon = load_pyg(name)
    # ensure undirected
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    # triangles
    if build_triangles:
        faces2 = triangles_from_edge_index(
            data.edge_index, data.num_nodes,
            use_cliques_if_small=True,
            degree_cap=degree_cap,
            max_tris=max_tris,
        )
    else:
        faces2 = np.zeros((0,3), dtype=int)

    # incidences & Laplacians
    B1, B2, E = build_B1_B2(data.edge_index, data.num_nodes, faces2)
    L0 = (B1 @ B1.T).tocsr()
    L1 = (B1.T @ B1 + B2 @ B2.T).tocsr()
    L2 = (B2.T @ B2).tocsr()

    out = {
        "name": canon,
        "num_nodes": data.num_nodes,
        "num_edges": E.shape[0],
        "num_tris": faces2.shape[0],
        "B1_csr": B1, "B2_csr": B2,
        "L0_csr": L0, "L1_csr": L1, "L2_csr": L2,
        "B1_t": scipy_to_torch(B1), "B2_t": scipy_to_torch(B2),
        "L0_t": scipy_to_torch(L0), "L1_t": scipy_to_torch(L1), "L2_t": scipy_to_torch(L2),
        "x": data.x, "y": data.y,
        "masks": { "train": getattr(data, "train_mask", None),
                   "val":   getattr(data, "val_mask", None),
                   "test":  getattr(data, "test_mask", None) },
        "faces2": faces2,  # (n_tri, 3) np.ndarray
        "edges": E,        # (n_edge, 2) np.ndarray (u<v)
    }
    return out

# ---- Examples ----
cora = build_simplicial_from_pyg("cora")  # small: builds all triangles
pubmed = build_simplicial_from_pyg("pubmed", degree_cap=60, max_tris=1_000_000)  # guardrails
flickr = build_simplicial_from_pyg("flickr", degree_cap=40, max_tris=500_000)    # strong caps
for pack in [cora, pubmed, flickr]:
    print(pack["name"], "#V", pack["num_nodes"], "#E", pack["num_edges"], "#Δ", pack["num_tris"])