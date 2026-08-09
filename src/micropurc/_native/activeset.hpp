// Active-set support: DAG/connectivity checks and kept-row grounding.
#pragma once

#include <cstdint>
#include <vector>

// Check the DAG property and reverse reachability of dest from explicit active
// edge lists. is_dag: no directed cycles. is_connected: every active node
// reaches dest via reverse edges. Both are out-parameters and are always
// written.
void check_dag_connectivity_edges(const int *active_tail, const int *active_head, int n_active,
                                  int dest, int V, bool &is_dag, bool &is_connected);

// Same check, but the active edges are given as indices into the tail/head
// arrays of the full network.
void check_dag_connectivity(const int *tail, const int *head, int n_active, const int *active_idx,
                            int dest, int V, bool &is_dag, bool &is_connected);

// Component-aware (pseudoinverse-faithful) grounding: returns the node rows of
// A to keep, one row per connected component of the active subgraph having been
// dropped so that A A^T over the kept rows is SPD. n_components_out reports how
// many components were found, hence how many rows were dropped.
void find_kept_rows_robust(const int *tail, const int *head, const int *active_idx, int n_active,
                           int dest, int V, const int64_t *A_indptr, const int64_t *A_indices,
                           std::vector<int> &kept_rows, int &n_components_out);

void find_kept_rows_robust_edges(const int *active_tail, const int *active_head, int n_active,
                                 int dest, int V, std::vector<int> &kept_rows,
                                 int &n_components_out);
