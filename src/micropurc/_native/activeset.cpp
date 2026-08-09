// Definitions for activeset.hpp: DAG/connectivity checks and kept-row grounding.

#include "activeset.hpp"

#include "common.hpp"

// Reports two independent properties of the active subgraph from one adjacency
// build:
//   is_dag       — no directed cycles, which the route walk needs to terminate.
//   is_connected — every active node reaches dest along active edges.
// An empty active edge list is vacuously both. A dest outside the subgraph
// short-circuits to is_connected = false with is_dag left at its optimistic
// true, so a caller that reads is_dag alone must handle that case itself.
void check_dag_connectivity_edges(const int *active_tail, const int *active_head, int n_active,
                                  int dest, int V, bool &is_dag, bool &is_connected) {
    if (n_active == 0) {
        is_dag = true;
        is_connected = true;
        return;
    }

    std::vector<std::vector<int>> fwd(V), rev(V);
    std::vector<bool> in_graph(V, false);

    for (int i = 0; i < n_active; i++) {
        int t = active_tail[i], h = active_head[i];
        fwd[t].push_back(h);
        rev[h].push_back(t);
        in_graph[t] = true;
        in_graph[h] = true;
    }

    is_dag = true;
    is_connected = false;

    if (!in_graph[dest]) return;

    // DAG check: iterative DFS with 3 colours (0=unseen, 1=on the current
    // stack, 2=finished). Reaching a gray node closes a cycle.
    std::vector<int> color(V, 0);
    std::vector<int> stack;

    for (int v = 0; v < V && is_dag; v++) {
        if (!in_graph[v] || color[v] != 0) continue;
        stack.push_back(v);
        while (!stack.empty() && is_dag) {
            int u = stack.back();
            if (color[u] == 0) {
                color[u] = 1;
                for (int w : fwd[u]) {
                    if (color[w] == 1) {
                        is_dag = false;
                        break;
                    }
                    if (color[w] == 0) stack.push_back(w);
                }
            } else {
                stack.pop_back();
                color[u] = 2;
            }
        }
    }

    // Reverse reachability from dest.
    std::vector<bool> reached(V, false);
    stack.clear();
    stack.push_back(dest);
    reached[dest] = true;
    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        for (int w : rev[u]) {
            if (!reached[w]) {
                reached[w] = true;
                stack.push_back(w);
            }
        }
    }

    is_connected = true;
    for (int v = 0; v < V; v++) {
        if (in_graph[v] && !reached[v]) {
            is_connected = false;
            break;
        }
    }
}

void check_dag_connectivity(const int *tail, const int *head, int n_active, const int *active_idx,
                            int dest, int V, bool &is_dag, bool &is_connected) {
    std::vector<int> active_tail(n_active), active_head(n_active);
    for (int i = 0; i < n_active; i++) {
        int e = active_idx[i];
        active_tail[i] = tail[e];
        active_head[i] = head[e];
    }
    check_dag_connectivity_edges(active_tail.data(), active_head.data(), n_active, dest, V, is_dag,
                                 is_connected);
}

// Component-aware grounding (robust_pinv). The active incidence has one
// redundant node row per connected component: within a component the rows sum
// to zero, so A A^T is singular until one row per component is dropped. Drop
// exactly one and the remaining rows are independent, which makes the Schur
// system SPD without any regularization. Dest's component drops dest, so the
// retained potentials are measured against the destination; every other
// component drops its lowest-numbered node, which makes the choice
// deterministic and independent of edge order.
void find_kept_rows_robust(const int *tail, const int *head, const int *active_idx, int n_active,
                           int dest, int V, const int64_t *A_indptr, const int64_t *A_indices,
                           std::vector<int> &kept_rows, int &n_components_out) {
    // Find nodes in the active subgraph.
    std::vector<bool> in_graph(V, false);
    for (int i = 0; i < n_active; i++) {
        int e = active_idx[i];
        in_graph[tail[e]] = true;
        in_graph[head[e]] = true;
    }

    // Map active nodes to dense indices for union-find.
    std::vector<int> node_to_dense(V, -1);
    std::vector<int> dense_to_node;
    for (int v = 0; v < V; v++) {
        if (in_graph[v]) {
            node_to_dense[v] = (int)dense_to_node.size();
            dense_to_node.push_back(v);
        }
    }
    int n_nodes = (int)dense_to_node.size();
    if (n_nodes == 0) {
        kept_rows.clear();
        n_components_out = 0;
        return;
    }

    UnionFind uf(n_nodes);
    for (int i = 0; i < n_active; i++) {
        int e = active_idx[i];
        uf.unite(node_to_dense[tail[e]], node_to_dense[head[e]]);
    }
    n_components_out = uf.n_components;

    // One dropped node per component, keyed by union-find root.
    std::vector<int> comp_dropped(n_nodes, -1);  // root → node to drop
    int dest_dense = node_to_dense[dest];

    for (int i = 0; i < n_nodes; i++) {
        int root = uf.find(i);
        int node = dense_to_node[i];
        if (comp_dropped[root] < 0) {
            if (dest_dense >= 0 && uf.find(dest_dense) == root) {
                comp_dropped[root] = dest;
            } else {
                comp_dropped[root] = node;
            }
        } else if (dest_dense >= 0 && uf.find(dest_dense) == root) {
            // Dest's component: always drop dest.
            comp_dropped[root] = dest;
        } else {
            // Other component: drop minimum node.
            comp_dropped[root] = std::min(comp_dropped[root], node);
        }
    }

    // Build the set of nodes to drop.
    std::vector<bool> drop(V, false);
    for (int i = 0; i < n_nodes; i++) {
        int root = uf.find(i);
        drop[comp_dropped[root]] = true;
    }

    // Kept rows are the node rows A actually touches on the active columns, less
    // the dropped ones. Reading A directly keeps this correct for any incidence
    // layout the caller supplies.
    std::vector<bool> has_entry(V, false);
    for (int i = 0; i < n_active; i++) {
        int col = active_idx[i];
        for (int64_t p = A_indptr[col]; p < A_indptr[col + 1]; p++) {
            has_entry[A_indices[p]] = true;
        }
    }

    kept_rows.clear();
    for (int r = 0; r < V; r++) {
        if (has_entry[r] && !drop[r]) kept_rows.push_back(r);
    }
}

// Same grounding rule as find_kept_rows_robust, for callers that hold the active
// edges as explicit (tail, head) pairs and have no CSC incidence at hand. Node
// membership comes from those endpoints, so a dest outside [0, V) leaves every
// component grounded at its lowest-numbered node.
void find_kept_rows_robust_edges(const int *active_tail, const int *active_head, int n_active,
                                 int dest, int V, std::vector<int> &kept_rows,
                                 int &n_components_out) {
    std::vector<bool> in_graph(V, false);
    for (int i = 0; i < n_active; i++) {
        in_graph[active_tail[i]] = true;
        in_graph[active_head[i]] = true;
    }

    std::vector<int> node_to_dense(V, -1);
    std::vector<int> dense_to_node;
    for (int v = 0; v < V; v++) {
        if (in_graph[v]) {
            node_to_dense[v] = (int)dense_to_node.size();
            dense_to_node.push_back(v);
        }
    }
    int n_nodes = (int)dense_to_node.size();
    if (n_nodes == 0) {
        kept_rows.clear();
        n_components_out = 0;
        return;
    }

    UnionFind uf(n_nodes);
    for (int i = 0; i < n_active; i++) {
        uf.unite(node_to_dense[active_tail[i]], node_to_dense[active_head[i]]);
    }
    n_components_out = uf.n_components;

    std::vector<int> comp_dropped(n_nodes, -1);
    int dest_dense = (dest >= 0 && dest < V) ? node_to_dense[dest] : -1;

    for (int i = 0; i < n_nodes; i++) {
        int root = uf.find(i);
        int node = dense_to_node[i];
        if (comp_dropped[root] < 0) {
            if (dest_dense >= 0 && uf.find(dest_dense) == root) {
                comp_dropped[root] = dest;
            } else {
                comp_dropped[root] = node;
            }
        } else if (dest_dense >= 0 && uf.find(dest_dense) == root) {
            comp_dropped[root] = dest;
        } else {
            comp_dropped[root] = std::min(comp_dropped[root], node);
        }
    }

    std::vector<bool> drop(V, false);
    for (int i = 0; i < n_nodes; i++) {
        int root = uf.find(i);
        drop[comp_dropped[root]] = true;
    }

    kept_rows.clear();
    for (int r = 0; r < V; r++) {
        if (in_graph[r] && !drop[r]) kept_rows.push_back(r);
    }
}
