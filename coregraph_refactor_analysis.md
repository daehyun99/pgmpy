# `_CoreGraph` Refactor — Status

**Branch:** `Refactor/2385/PDAG-1` · **Re-evaluated:** 2026-06-07 (PDAG/ADMG/MAG reverted to standalone dev classes)

`_CoreGraph` (marker-based; `_base.py` / `_algorithms.py` / `_plotting.py`) is the **target** unified base. It is currently **inherited by no production class** — `DAG`, `PDAG`, `ADMG`, `MAG` each keep their own dev representation and only share `_GraphRolesMixin`. The refactor goal is to migrate them onto `_CoreGraph`.

## 1. Done — `_CoreGraph` base built (not yet adopted)

- **Edges / structure:** `add_edge`, `add_edges_from`, `remove_edge`, `remove_edges_from`, `replace_edge`, `has_edge`, `get_edge`, `get_edges` (`edge_types` filter + canonical orientation), `get_unique_edge_types`.
- **Traversal:** `get_neighbors` / `get_reachable_nodes` (single type or collection of `edge_types`), `get_parents` / `get_children` / `get_spouses` / `get_ancestors` / `get_descendants`, `get_roots` / `get_leaves`, `has_path` / `get_all_paths` / `has_direct_path`, `is_collider` / `has_inducing_path`, `has_directed_cycle` / `has_almost_directed_cycle`, `get_topological_order`, `get_markov_blanket` / `get_ancestral_graph`.
- **Transforms / export:** `get_subgraph`, `get_skeleton`, `do`, `copy`, `to_adjacency`, `to_graphviz` / `to_daft`, `__eq__` / `__hash__`.

Only a test fixture (`_DirectedGraph(_CoreGraph)`) subclasses it today.

## 2. Production classes to migrate onto `_CoreGraph`

Each has its own representation; the third column is bespoke code that an `_CoreGraph` migration would **replace** with the shared method, the fourth is class-specific algorithms with **no** `_CoreGraph` equivalent yet.

| Class | Base / representation | Maps to existing `_CoreGraph` method | Class-specific (no equivalent) |
|---|---|---|---|
| **DAG** | `_GraphRolesMixin, nx.DiGraph` | get_parents/children/ancestors/descendants (list→set), copy, `__eq__` | moralize, get_immoralities, to_pdag, edge_strength, get_stats |
| **PDAG** | `_GraphRolesMixin, nx.DiGraph`; `directed_edges` / `undirected_edges` sets (undirected stored both ways) | directed_parents/children→get_parents/children, undirected_neighbors→`get_neighbors("--")`, all_neighbors→get_neighbors, has_directed/undirected_edge & is_adjacent→has_edge, copy, to_graphviz, `__eq__` | chain_component, is_clique, has_semidirected_path (≈ `has_path` over `->`+`--`), orient_undirected_edge, apply_meeks_rules, has_acyclic_extension, calibrate_directed_undirected_edges, to_dag, to_cpdag |
| **ADMG** | `_GraphRolesMixin, MultiDiGraph`; edge `type="directed"/"bidirected"` | get_directed_parents→get_parents, get_bidirected_parents & get_spouses→get_spouses, get_children/ancestors/descendants, get_ancestral_graph, get_markov_blanket, `__eq__`; add_directed/bidirected_edges→`add_edge(edge_type)` | get_district (≈ reachable via `<>`), to_dag (latent projection), is_mseparated / is_mconnected / mconnected_nodes |
| **MAG** | `AncestralBase(nx.Graph, _GraphRolesMixin)`; edge `marks` attr, 4-tuple ebunch | _is_collider→is_collider, has_inducing_path, AncestralBase get_neighbors(u_type/v_type) / get_reachable_nodes / get_ancestors → `get_*(edge_types)` | is_visible_edge, lower_manipulation, upper_manipulation |

## 3. Gaps — what `_CoreGraph` still lacks

- **Generic, worth centralizing:** `is_clique`, `chain_component`, `get_simple_cycles`, the m-separation family (`is_m_separator`, … — commented-out stubs in `_algorithms.py`), `moralize`, `get_immoralities`.
- **Class-specific algorithms** (decide per class: move onto `_CoreGraph` vs. keep as a subclass override): Meek's rules + `to_dag` / `to_cpdag` (PDAG); latent-projection `to_dag`, districts, m-separation (ADMG); inducing/visible-path + lower/upper manipulation (MAG).
- **Then:** the ~30 external raw-`nx` call sites (`.predecessors`, `.neighbors`, `nx.topological_sort`, `.subgraph`, `nx.has_path`, … — full counts in git history of this file) can be routed to `_CoreGraph` methods only **after** the classes actually inherit `_CoreGraph`.

## Corrections vs. the previous version

- `_CoreGraph` is inherited by **no** production class (previously stated "inherited by MAG/ADMG/PDAG").
- `AncestralBase` is **not** orphaned — it is `MAG`'s base, so "delete AncestralBase" no longer applies.
- `chain_component` / `is_clique` / `has_semidirected_path` live on the **standalone dev PDAG** (a `nx.DiGraph`), not on the `_CoreGraph` layer.

---

*Deferred: should `__eq__` be structural-only (ignore edge/node attrs like `weight`/`strength`) so it + `__hash__` share one key? Today `__eq__` = full `graphs_equal`.*
