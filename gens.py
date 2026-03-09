def test_directed(edge_index):
    set_edges = set()
    bi_direct = 0
    self_loop = 0
    for i in range(edge_index.shape[1]):
        if edge_index[1][i].item() == edge_index[0][i].item():
            self_loop += 1
        edge_inv = frozenset([edge_index[1][i].item(), edge_index[0][i].item()])

        edge = frozenset([edge_index[0][i].item(), edge_index[1][i].item()])
        if edge_inv in set_edges:
            bi_direct += 1
        set_edges.add(edge)
    print("selfloop: {}, Num_bidirect_edges: {}, total_num_edges: {}".format(self_loop, bi_direct, edge_index.shape[1]))
    if bi_direct * 2 == edge_index.shape[1] - self_loop:
        return False
    return True


