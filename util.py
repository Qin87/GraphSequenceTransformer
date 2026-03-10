import torch
from collections import deque, defaultdict

def build_adj(n, edges):
    adj = defaultdict(list)
    for i in range(edges.shape[1]):
        u = edges[0, i].item()
        v = edges[1, i].item()
        adj[u].append(v)
        adj[v].append(u)
    return adj


def bfs_farthest(start, adj, n):
    visited = [False] * n
    dist = [-1] * n

    q = deque([start])
    visited[start] = True
    dist[start] = 0

    farthest_node = start

    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                dist[v] = dist[u] + 1
                q.append(v)

                if dist[v] > dist[farthest_node]:
                    farthest_node = v

    return farthest_node, dist[farthest_node]


def spanning_tree_adj(n, edges):
    adj = build_adj(n, edges)

    tree_adj = defaultdict(list)
    visited = [False] * n
    q = deque([0])
    visited[0] = True

    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_adj[u].append(v)
                tree_adj[v].append(u)
                q.append(v)

    return tree_adj


def spanning_tree_diameter(X, edges):
    n = X.shape[0]

    # build spanning tree
    tree_adj = spanning_tree_adj(n, edges)

    # first BFS
    u, _ = bfs_farthest(0, tree_adj, n)

    # second BFS
    v, diameter = bfs_farthest(u, tree_adj, n)

    return diameter

if __name__ == '__main__':
    # Example
    n = 6
    d = 3
    X = torch.randn(n, d)

    edges = torch.tensor([
        [0, 0, 1, 2, 3, 4],
        [1, 2, 3, 3, 4, 5]
    ])

    print(spanning_tree_diameter(X, edges))