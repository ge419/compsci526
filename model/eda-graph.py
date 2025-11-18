"""Build an ingredient co-occurrence graph.

Nodes are ingredient strings. Edges connect two ingredients that appear
together in the same recipe; the edge weight equals the number of recipes
where the pair co-occurs.

Outputs written to `--out-dir`:
 - ingredient_graph.graphml  (GraphML with `weight` edge attribute)
 - edge_list.csv             (u,v,weight)
 - ingredient_counts.csv     (ingredient,count)

Usage:
  python model/eda-graph.py            # auto-find local .arrow under model/dataset
  python model/eda-graph.py /path/to/mm-food-100_k-train.arrow
  python model/eda-graph.py --data-file path/to/file.csv

Requirements:
  pip install datasets pandas networkx tqdm
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import itertools
from collections import Counter
from pathlib import Path
import re

try:
	from datasets import Dataset, load_dataset
except Exception:
	Dataset = None
	load_dataset = None

try:
	import networkx as nx
except Exception:
	print("Missing dependency: networkx. Install with `pip install networkx`")
	raise

import pandas as pd
from tqdm import tqdm


def find_local_arrow(filename="mm-food-100_k-train.arrow") -> Path | None:
	root = Path(__file__).resolve().parents[1]
	# common location in this repo: model/dataset/**/mm-food-100_k-train.arrow
	for p in root.rglob(filename):
		return p
	return None


def normalize_ingredient(s: str) -> str:
	"""Normalize ingredient text by removing stray brackets/quotes/punctuation.

	Keeps letters, digits, spaces, hyphen, slash, ampersand and plus.
	"""
	if not isinstance(s, str):
		s = str(s)
	s = s.strip().lower()
	# strip enclosing brackets/quotes
	s = re.sub(r'^[\[\(\{<"\']+', '', s)
	s = re.sub(r'[\]\)\}\>"\']+$', '', s)
	# remove leading bullets/hyphens
	s = re.sub(r'^[\-\u2022\*\+\s]+', '', s)
	# keep only common safe characters
	s = re.sub(r'[^a-z0-9\s\-/&+]', ' ', s)
	s = re.sub(r'\s+', ' ', s).strip()
	return s


def extract_ingredients_from_record(rec: dict) -> list[str]:
	# prefer a field named 'ingredients'
	if isinstance(rec, dict):
		for key in ("ingredients", "ingredient_list", "ingredient", "ing"):
			if key in rec and rec[key]:
				val = rec[key]
				if isinstance(val, list):
					return [normalize_ingredient(x) for x in val if x and str(x).strip()]
				if isinstance(val, str):
					# split on comma/semicolon
					parts = [p.strip() for p in val.replace(";", ",").split(",")]
					parts = [p for p in parts if p]
					return [normalize_ingredient(x) for x in parts]
	# fallback: if record has any list-of-strings, use it
	if isinstance(rec, dict):
		for k, v in rec.items():
			if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
				return [normalize_ingredient(x) for x in v if x and str(x).strip()]
	return []


def build_graph_from_records(records, min_cooccurrence=1, top_k_ingredients=None):
	ing_counts = Counter()
	edge_counts = Counter()

	for rec in tqdm(records, desc="Processing recipes"):
		# datasets.arrow Dataset yields dict-like items; pandas records are dicts
		ings = extract_ingredients_from_record(rec)
		if not ings:
			continue
		uniq = sorted(set(ings))
		for ing in uniq:
			ing_counts[ing] += 1
		for a, b in itertools.combinations(uniq, 2):
			if a == b:
				continue
			edge = (a, b) if a < b else (b, a)
			edge_counts[edge] += 1

	if top_k_ingredients is not None:
		top_ings = set([ing for ing, _ in ing_counts.most_common(top_k_ingredients)])
	else:
		top_ings = set(ing_counts.keys())

	G = nx.Graph()
	for ing in top_ings:
		G.add_node(ing, count=ing_counts.get(ing, 0))

	for (a, b), w in edge_counts.items():
		if w < min_cooccurrence:
			continue
		if a not in top_ings or b not in top_ings:
			continue
		G.add_edge(a, b, weight=w)

	return G, ing_counts, edge_counts


def load_records_from_arrow(path: Path):
	if Dataset is None:
		raise RuntimeError("datasets package not available. Install with `pip install datasets`")
	ds = Dataset.from_file(str(path))
	# iterate over records (dict-like)
	return ds


def load_records_from_csv(path: Path):
	df = pd.read_csv(str(path))
	return df.to_dict(orient='records')


def main():
	p = argparse.ArgumentParser(description="Build ingredient co-occurrence graph")
	p.add_argument("data_file", nargs="?", help="Path to .arrow or CSV file (optional)")
	p.add_argument("--out-dir", default="data/graph", help="Output directory")
	p.add_argument("--min-cooccurrence", type=int, default=1)
	p.add_argument("--top-k-ingredients", type=int, default=None)
	args = p.parse_args()

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	records = None
	data_path = None
	if args.data_file:
		data_path = Path(args.data_file)
		if not data_path.exists():
			print(f"Data file {data_path} not found.")
			sys.exit(2)
		if data_path.suffix == ".arrow":
			records = load_records_from_arrow(data_path)
		elif data_path.suffix in (".csv", ".tsv"):
			records = load_records_from_csv(data_path)
		else:
			# try treating as JSONL
			with open(data_path, 'r', encoding='utf8') as f:
				recs = []
				for line in f:
					line = line.strip()
					if not line:
						continue
					try:
						recs.append(json.loads(line))
					except Exception:
						pass
				if recs:
					records = recs
	else:
		# try to find arrow file under model/dataset
		arrow = find_local_arrow()
		if arrow:
			print(f"Found local arrow: {arrow}")
			records = load_records_from_arrow(arrow)
			data_path = arrow
		else:
			print("No local .arrow file found. Provide path as first argument or use --data-file")
			sys.exit(2)

	G, ing_counts, edge_counts = build_graph_from_records(records, min_cooccurrence=args.min_cooccurrence, top_k_ingredients=args.top_k_ingredients)

	out_graphml = out_dir / 'ingredient_graph.graphml'
	out_edges = out_dir / 'edge_list.csv'
	out_freq = out_dir / 'ingredient_counts.csv'

	print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()} — writing to {out_dir}")
	nx.write_graphml(G, out_graphml)

	# edge list
	rows = []
	for u, v, d in G.edges(data=True):
		rows.append((u, v, d.get('weight', 1)))
	pd.DataFrame(rows, columns=['u','v','weight']).to_csv(out_edges, index=False)

	pd.DataFrame(list(ing_counts.items()), columns=['ingredient','count']).sort_values('count', ascending=False).to_csv(out_freq, index=False)

	print("Done.")


if __name__ == '__main__':
	main()

