# `_CoreGraph` Refactor Analysis — Shared Methods & Raw-Operation Audit

**Branch:** `Refactor/2385/PDAG-1`
**Date:** 2026-06-05
**Scope:** `pgmpy/base/_base.py` (`_CoreGraph`), `pgmpy/base/_mixin_algorithms.py` (`_GraphAlgorithmMixin`), and the graph classes `DAG`, `PDAG`, `ADMG`, `MAG` (+ `AncestralBase`), plus a codebase-wide audit of raw networkx operations on graph instances.

This document answers two questions:

1. Which methods should be **added** to `_CoreGraph` / `_GraphAlgorithmMixin` so they are shared across all graph classes?
2. Where in the codebase are we performing **raw networkx operations** on `DAG`/`MAG`/`ADMG`/`PDAG` instances that should instead go through encapsulated methods?

---

## Part 1 — Methods to add to the shared layer

**Framing:** most "candidates" surfaced by the analysis *already exist* on `_CoreGraph`/`_GraphAlgorithmMixin` — they are merely re-implemented in `DAG` (not yet migrated to `_CoreGraph`) and in `AncestralBase` (orphaned legacy). So the genuinely **new** shared methods are a smaller set. The lists below separate "already exists → migrate/route to it" from "actually add".

### 1a. Already shared — reconcile & route to them (do **not** re-add)

These exist on `_CoreGraph` / `_GraphAlgorithmMixin` and are duplicated by `DAG` and `AncestralBase`:

`get_parents`, `get_children`, `get_spouses`, `get_neighbors`, `get_ancestors`, `get_descendants`, `get_reachable_nodes`, `get_markov_blanket`, `get_ancestral_graph`, `copy`, `__eq__`, `has_direct_path`, `has_directed_cycle`, `is_collider`, `get_directed_subgraph`.

- **Central caveat:** `DAG`'s versions return **lists** (and accept node-or-iterable); `_CoreGraph`'s return **sets** (single node, includes the node itself for ancestor/descendant). Reconciling **list → set** is the main migration hazard.
- **`AncestralBase` is fully orphaned legacy** — its 11 traversal/copy/eq methods all duplicate `_CoreGraph`. It is exported and tested but inherited by nothing (`MAG`/`ADMG` subclass `_CoreGraph`, not `AncestralBase`). Recommend deletion once `MAG`/`ADMG` independence is confirmed.

### 1b. NEW methods to add to `_CoreGraph` (structural / data-level)

| Method | Why | Evidence |
|---|---|---|
| `get_topological_order()` | wraps `nx.topological_sort` | 10 raw-op sites (LinearGaussian / Functional BN) |
| `to_pandas_adjacency()` (or `adjacency_matrix`) | every score-based learner rebuilds this | 11 sites: ChowLiu, TAN, PC, GES, HillClimb, TOPIC |
| `subgraph(nodes)` / `get_induced_subgraph(nodes)` | class-preserving node-induced subgraph (raw `.subgraph` drops type/CPDs) | 6 sites + `DAG.get_stats` / `get_ancestral_graph` |
| `get_leaves()` / `get_roots()` | sink / source nodes (currently DAG-only via `in/out_degree`) | `DAG.py:491/510` |
| `to_undirected()` / `get_skeleton()` | undirected skeleton | used by `moralize`, `is_iequivalent` |
| `to_graphviz()` | identical implementation in `DAG` and `PDAG` | hoist the duplicate |
| `__hash__` | `_CoreGraph` defines `__eq__` but no `__hash__` (DAG has one) | hashability |

### 1c. NEW methods to add to `_GraphAlgorithmMixin` (algorithm-level)

| Method | Why | Evidence |
|---|---|---|
| `moralize()` | moral graph (skeleton + marry parents) — generic | DAG-only today |
| `get_immoralities()` (v-structures) | generic for PDAG/ADMG too; mixin has `is_collider` but no enumerator | DAG-only |
| `has_semidirected_path()` | hoist from PDAG (its body is currently dead-code trapped in a docstring) | PDAG |
| `chain_component()` | undirected reachability; built on `get_reachable_nodes('--')` | PDAG |
| `is_clique()` | **duplicated verbatim** in PDAG and `UndirectedGraph` | PDAG / UndirectedGraph |
| `get_simple_paths(u, v)` | wraps `nx.all_simple_paths` | 6 sites: HillClimb, CausalInference, frontdoor, adjustment |
| `get_simple_cycles()` / `find_cycles()` | wraps `nx.simple_cycles` | ExpertInLoop, expert |
| m-separation family | `is_m_separator`, `is_m_connected`, `get_m_separator(s)`, … are **commented-out stubs** in the mixin | `_mixin_algorithms.py` |
| `do(nodes)` (borderline) | intervention edge-surgery; generic | DAG-only |

---

## Part 2 — Raw networkx operations on graph classes

**Yes — extensively. 108 confirmed sites.** The dominant root cause is that **`DAG` is not migrated to `_CoreGraph`**, so callers reach for raw `nx` (and `DAG.get_parents` etc. are themselves raw `nx` wrappers returning lists). The findings split into three buckets.

### 2a. Route to a method that already exists (~38 sites)

| Raw op | Count | → existing method |
|---|---|---|
| `g.predecessors(x)` | 18 | `g.get_parents(x)` |
| `g.neighbors(x)` | 11 | `g.get_neighbors(x)` |
| `g.successors(x)` | 6 | `g.get_children(x)` |
| `nx.descendants(g, x)` | 4 | `g.get_descendants(x)` |
| `nx.ancestors(g, x)` | 2 | `g.get_ancestors(x)` |
| `nx.is_directed_acyclic_graph(g)` / `nx.find_cycle(g)` | 4 | `g.has_directed_cycle()` |
| `nx.has_path(g, u, v)` (directed) | 5 | `g.has_direct_path(u, v)` |

Hotspots: `inference/CausalInference.py` (~12 sites), `estimators/EM.py`, `estimators/StructureScore.py`, `models/DiscreteBayesianNetwork.py`, `causal_discovery/_base.py`.

### 2b. Need a NEW shared method (the Part 1 list) (~36 sites)

| Raw op | Count | → new method |
|---|---|---|
| `nx.to_pandas_adjacency(g)` | 11 | `g.to_pandas_adjacency()` |
| `nx.topological_sort(g)` | 10 | `g.get_topological_order()` |
| `nx.all_simple_paths(g, …)` | 6 | `g.get_simple_paths(u, v)` |
| `g.subgraph(nodes)` / `nx.DiGraph(g.subgraph(...))` | 6 | `g.subgraph(nodes)` (class-preserving) |
| `g.in_degree` / `g.out_degree` | 6 | `get_roots()` / `get_leaves()` |
| `g.to_undirected()` | 3 | `g.to_undirected()` / reuse `moralize()` |
| `nx.simple_cycles(g)` | 2 | `g.get_simple_cycles()` |
| `nx.DiGraph(g.edges())` (`shd.py`), `g.to_directed()`, `g.edges[u, v]` strength | 3 | `get_directed_subgraph()` / edge-attr accessors |

### 2c. Out of strict DAG-family scope (~10 sites)

Raw `.neighbors` / `.subgraph` / BFS on **`JunctionTree`** (`inference/ExactInference.py`) and **`UndirectedGraph`** (`inference/EliminationOrder.py`) — these subclass `nx.Graph` directly, not `_CoreGraph`. Worth encapsulating, but separate from the `_CoreGraph` family.

### 2d. Raw ops *inside* the graph classes themselves

- `DAG.py`: ~15 internal raw-`nx` sites — `get_stats`, `active_trail_nodes`, `get_leaves`/`get_roots`, `moralize`, `_check_cycles`, `is_iequivalent`, `edge_strength`.
- `PDAG.has_acyclic_extension`: builds `nx.DiGraph(self.edges())`.

These are precisely the methods that should *become* the shared abstractions in Part 1.

---

## Bottom line / recommended sequencing

1. **Migrate `DAG` onto `_CoreGraph`** — the single biggest lever. Instantly resolves ~38 "route-to-existing" raw-op sites and removes the list-vs-set fork.
2. **Add the ~7 new `_CoreGraph` methods** (`get_topological_order`, `to_pandas_adjacency`, `subgraph`, `get_leaves`/`get_roots`, `to_undirected`/`get_skeleton`, `to_graphviz`, `__hash__`).
3. **Add the ~8 new `_GraphAlgorithmMixin` methods** (`moralize`, `get_immoralities`, `has_semidirected_path`, `chain_component`, `is_clique`, `get_simple_paths`, `get_simple_cycles`, m-separation family).
4. **Delete `AncestralBase`** (orphaned; 11 duplicate methods) once `MAG`/`ADMG` independence is confirmed.

A good first increment that does **not** depend on the DAG migration: add the high-traffic `_CoreGraph` methods `get_topological_order`, `to_pandas_adjacency`, and `subgraph` (with tests) — they unblock the most call sites.
