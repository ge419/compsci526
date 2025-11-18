"""Filter an ingredient graph to top-N ingredients and cluster nodes by community.

This script:
- Loads a GraphML (default or one you pass)
- Optionally loads ingredient frequencies (`ingredient_counts_filtered.csv`) to pick top-N
- Filters nodes to top-N by frequency (or you can pass --min-degree)
- Runs community detection (greedy modularity) to cluster nodes
- Writes clustered GraphML and a PNG visualization colored by cluster

Usage example:
  python tools/filter_and_cluster_graph.py \
    --graph data/graph_filtered/ingredient_graph_filtered.graphml \
    --counts data/graph_filtered/ingredient_counts_filtered.csv \
    --top-k 500 \
    --out data/graph_clustered \
    --use-forceatlas --gravity 1.0 --fa2-iterations 200

Requirements: networkx, matplotlib, pandas
Optional: fa2 / fa2-modified (ForceAtlas2) for a layout with a gravity parameter:
  pip install fa2-modified
"""
from __future__ import annotations

import argparse
from pathlib import Path
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import math
from networkx.algorithms import community as nx_comm
import warnings


def load_counts(counts_path: Path):
    df = pd.read_csv(counts_path)
    # expect columns: ingredient, count
    return {row['ingredient']: int(row['count']) for _, row in df.iterrows()}


def filter_graph_by_topk(G: nx.Graph, counts: dict, top_k: int):
    # choose top_k ingredients that are present in G
    items = [(ing, counts.get(ing, 0)) for ing in G.nodes()]
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    keep = set([ing for ing, _ in items_sorted[:top_k]])
    return G.subgraph(keep).copy()


def filter_graph_by_min_degree(G: nx.Graph, min_degree: int):
    keep = [n for n, d in G.degree() if d >= min_degree]
    return G.subgraph(keep).copy()


def cluster_graph(G: nx.Graph):
    # greedy modularity communities -> returns list of sets
    communities = list(nx_comm.greedy_modularity_communities(G, weight='weight'))
    # assign cluster id per node
    node_to_cluster = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            node_to_cluster[n] = cid
    return node_to_cluster, len(communities)


def _prepare_spring_weights(G: nx.Graph, weight_attr: str = 'weight'):
    """Attach 'spring_w' attributes to edges using log1p compression of the weight attribute."""
    for u, v, d in G.edges(data=True):
        try:
            w = float(d.get(weight_attr, 1.0))
        except Exception:
            w = 1.0
        d['spring_w'] = math.log1p(max(0.0, w))


def _try_import_forceatlas():
    """
    Try to import ForceAtlas2 from several possible package names/variants.
    Returns the ForceAtlas2 class or raises ImportError.
    """
    import importlib

    candidates = [
        ('fa2', 'ForceAtlas2'),
        ('fa2_modified', 'ForceAtlas2'),
        ('fa2-modified', 'ForceAtlas2'),  # unlikely module name, but harmless to try
    ]
    for module_name, cls_name in candidates:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, cls_name):
                    return getattr(module, cls_name)
            except Exception:
                # try next candidate if importing failed
                continue
    # final attempt: try 'fa2' import directly (covers some installs)
    try:
        from fa2 import ForceAtlas2  # type: ignore
        return ForceAtlas2
    except Exception:
        pass
    # try maintained fork name
    try:
        from fa2_modified import ForceAtlas2  # type: ignore
        return ForceAtlas2
    except Exception:
        pass

    raise ImportError("ForceAtlas2 (fa2 / fa2-modified / fa2_modified) not found")


def draw_clustered(
    G: nx.Graph,
    node_to_cluster: dict,
    out_png: Path,
    title: str = None,
    gravity: float = 0.0,
    use_forceatlas: bool = False,
    fa2_iterations: int = 200,
):
    """
    Draw the clustered graph to PNG.

    Parameters:
    - gravity: 0.0 = no centripetal pull. Larger values increase pull toward center.
    - use_forceatlas: if True, try to use ForceAtlas2 (fa2) which supports a gravity parameter.
                      If fa2 is unavailable it falls back to the center-node hack.
    - fa2_iterations: iterations for ForceAtlas2 if used.
    """
    plt.figure(figsize=(14, 12), facecolor='white')

    if G.number_of_nodes() == 0:
        warnings.warn("Graph has no nodes; nothing to draw.")
        plt.title("Empty graph")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return

    pos = None

    # Option A: Use ForceAtlas2 if requested and available
    if use_forceatlas:
        try:
            ForceAtlas2 = _try_import_forceatlas()
            # Ensure numeric weights exist
            for u, v, d in G.edges(data=True):
                try:
                    d['weight'] = float(d.get('weight', 1.0))
                except Exception:
                    d['weight'] = 1.0

            # Instantiate ForceAtlas2 with gravity param
            fa2 = ForceAtlas2(
                outboundAttractionDistribution=False,
                linLogMode=False,
                adjustSizes=False,
                edgeWeightInfluence=1.0,
                jitterTolerance=1.0,
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=max(0.0, float(gravity)),
                verbose=False,
            )
            # compute layout
            pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=fa2_iterations)
        except Exception as e:
            warnings.warn(f"ForceAtlas2 requested but unavailable/failed ({e}). Falling back to spring-layout gravity hack.")
            pos = None

    # Option B: Use spring_layout, possibly with a center-node gravity hack
    if pos is None:
        if gravity and gravity > 0.0:
            # Center-node hack: add a virtual center node connected to every node with small weight
            Gc = G.copy()
            center_node = "__CENTER__"
            degrees = [d for _, d in Gc.degree()] or [1]
            avg_deg = sum(degrees) / len(degrees) if degrees else 1.0
            # central_edge_weight controls how strongly nodes are pulled; scaled by avg degree and gravity
            central_edge_weight = float(gravity) * max(1.0, avg_deg)
            Gc.add_node(center_node)
            for n in list(Gc.nodes()):
                if n == center_node:
                    continue
                if not Gc.has_edge(n, center_node):
                    Gc.add_edge(n, center_node, weight=central_edge_weight)
                else:
                    Gc[n][center_node]['weight'] = Gc[n][center_node].get('weight', 0.0) + central_edge_weight

            _prepare_spring_weights(Gc, weight_attr='weight')
            try:
                pos_all = nx.spring_layout(Gc, weight='spring_w', seed=42, iterations=300)
                pos_all.pop(center_node, None)
                pos = pos_all
            except Exception:
                pos = nx.random_layout(G, seed=42)
        else:
            _prepare_spring_weights(G, weight_attr='weight')
            try:
                pos = nx.spring_layout(G, weight='spring_w', seed=42, iterations=300)
            except Exception:
                pos = nx.random_layout(G, seed=42)

    # Build clusters mapping
    clusters = {}
    for n, cid in node_to_cluster.items():
        clusters.setdefault(cid, []).append(n)

    cmap = plt.get_cmap('tab20')
    for cid, nodes in clusters.items():
        color = cmap(cid % 20)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=150, node_color=[color], label=f'c{cid}', alpha=0.9)

    # edges: use log-scaled widths capped to avoid overpainting
    edge_weights = [d.get('weight', 1) for _, _, d in G.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1.0
    widths = []
    for w in edge_weights:
        scaled = 0.2 + 6.0 * (math.log1p(w) / math.log1p(max_w)) if max_w > 0 else 0.2
        scaled = min(scaled, 8.0)
        widths.append(scaled)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.15)

    plt.title(title or 'Clustered ingredient graph')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', required=True)
    p.add_argument('--counts', required=False, help='CSV with ingredient,count')
    p.add_argument('--top-k', type=int, default=500)
    p.add_argument('--min-degree', type=int, default=None)
    p.add_argument('--min-edge-weight', type=float, default=0.0, help='Minimum edge weight to keep')
    p.add_argument('--out', default='data/graph_clustered')
    p.add_argument('--gravity', type=float, default=0.0, help='Centripetal gravity strength (0 = none)')
    p.add_argument('--use-forceatlas', action='store_true', help='Prefer ForceAtlas2 layout (fa2 / fa2-modified) if available')
    p.add_argument('--fa2-iterations', type=int, default=200, help='Iterations for ForceAtlas2 layout (if used)')
    args = p.parse_args()

    gpath = Path(args.graph)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading graph {gpath}...')
    G = nx.read_graphml(str(gpath))

    # ensure weights are numeric
    for u, v, d in G.edges(data=True):
        try:
            d['weight'] = float(d.get('weight', 1))
        except Exception:
            d['weight'] = 1.0

    if args.counts:
        counts = load_counts(Path(args.counts))
        Gf = filter_graph_by_topk(G, counts, args.top_k)
        print(f'Filtered to top-{args.top_k}: nodes {Gf.number_of_nodes()}, edges {Gf.number_of_edges()}')
    elif args.min_degree is not None:
        Gf = filter_graph_by_min_degree(G, args.min_degree)
        print(f'Filtered by min-degree {args.min_degree}: nodes {Gf.number_of_nodes()}, edges {Gf.number_of_edges()}')
    else:
        Gf = G

    # Filter edges by weight if requested
    if args.min_edge_weight and args.min_edge_weight > 0.0:
        remove = [(u, v) for u, v, d in Gf.edges(data=True) if d.get('weight', 0) < args.min_edge_weight]
        Gf.remove_edges_from(remove)
        # remove isolated nodes
        isolates = list(nx.isolates(Gf))
        if isolates:
            Gf.remove_nodes_from(isolates)
        print(f'Applied min-edge-weight {args.min_edge_weight}: nodes {Gf.number_of_nodes()}, edges {Gf.number_of_edges()}')

    node_to_cluster, ncomms = cluster_graph(Gf)
    print(f'Found {ncomms} communities')

    # annotate and write graphml
    for n in Gf.nodes():
        Gf.nodes[n]['cluster'] = node_to_cluster.get(n, -1)

    out_graphml = out_dir / f'ingredient_graph_clustered_top{args.top_k}.graphml'
    nx.write_graphml(Gf, out_graphml)

    out_png = out_dir / f'ingredient_graph_clustered_top{args.top_k}.png'
    draw_clustered(
        Gf,
        node_to_cluster,
        out_png,
        title=f'Clusters (top-{args.top_k})',
        gravity=args.gravity,
        use_forceatlas=args.use_forceatlas,
        fa2_iterations=args.fa2_iterations,
    )

    print('Wrote:', out_graphml, out_png)


if __name__ == '__main__':
    main()
