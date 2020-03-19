# _*_ coding::utf-8 _*_


from sklearn.neighbors import NearestNeighbors
from utils import *


def prepare_hyper_nodes(args, vani_adj):
    '''
    using BFS to analyze the sub-graphs in the vanilla graph,
    return
        the count of sub-graphs in the vanilla graph,
        the size of each sub-graph,
        the id set of each sub-graph.
    '''
    hyper_nodes = []
    for index, per_graph in enumerate(vani_adj):
        def bfs_graph(Graph):
            node_num = Graph[3]
            visited = [False for i in range(node_num)]
            neighbor = [[] for i in range(node_num)]

            for u, v in zip(list(Graph[0]), list(Graph[1])):
                neighbor[u].append(v)
                neighbor[v].append(u)

            # get the size of graph
            def bfs_neighbor(start_id):
                if visited[start_id]: return []
                Q = [start_id]
                visited[start_id] = True
                all_visited = [start_id]

                while len(Q) > 0:
                    nQ = []
                    for u in Q:
                        for v in neighbor[u]:
                            if visited[v]: continue
                            visited[v] = True
                            all_visited.append(v)
                            nQ.append(v)
                    Q = nQ
                return all_visited

            subgraph_num = 0
            subgraph_size = []
            subgraph_ids = []

            for i in range(node_num):
                if visited[i]: continue
                tmp_visited = bfs_neighbor(i)
                if len(tmp_visited) < args.minimum_subgraph_size: continue
                subgraph_num += 1
                subgraph_size.append(len(tmp_visited))
                subgraph_ids.append(tmp_visited)
            return subgraph_num, subgraph_size, subgraph_ids
        graph_num, graph_size, graph_ids = bfs_graph(per_graph)
        hyper_nodes.append([graph_num, graph_size, graph_ids])
    return hyper_nodes


def establish(args, vani_adjs, vani_ftr, labels, y_train, y_test, y_val, train_mask, test_mask, val_mask):
    '''
    there are many sub-graphs in vanilla graph, a hyper node represents a sub-graph.
    the meaning features of each sub-graph are used as the features of hyper nodes.
    and the hyper node's nearest neighbors are connected between each hyper nodes.
    '''
    num_supports = len(vani_adjs)                         # different kinds of graph
    hyper_nodes = prepare_hyper_nodes(args, vani_adjs)    # hyper_nodes: [[graph_num, graph_size, graph_ids], ...]
    normal_node_num = len(vani_ftr)                       # nodes count without hyper nodes
    hyper_node_num = sum([per[0] for per in hyper_nodes])
    total_num = normal_node_num + hyper_node_num
    vani_ftr_np = np.array(vani_ftr)

    edge_name = ["e%d" % i for i in range(1, num_supports + 1)]

    # using the meaning feature as the feature of hyper node
    features = vani_ftr.copy()
    for i in range(num_supports):
        for j in range(hyper_nodes[i][0]):  # hyper_nodes[i][0] : sub-graph count in vanilla graph
            features.append(list(np.mean(vani_ftr_np[hyper_nodes[i][2][j]], axis=0)))

    # finding K nearest neighbors of each hyper nodes
    whole_support = []
    pre_graph_sum = [0 for _ in range(num_supports)]
    for i in range(1, num_supports):
        pre_graph_sum[i] = pre_graph_sum[i - 1] + hyper_nodes[i - 1][0]

    features_np = np.array(features)
    for i in range(num_supports):
        st = normal_node_num + pre_graph_sum[i]
        ed = st + hyper_nodes[i][0]
        hyper_features = features_np[st:ed]
        clf = NearestNeighbors(n_neighbors=args.nearest_neighbor_K + 1, algorithm='ball_tree').fit(hyper_features)
        distances, indices = clf.kneighbors(hyper_features)

        adjs = vani_adjs[i].copy()
        for index, per in enumerate(indices):
            u = normal_node_num + pre_graph_sum[i] + index
            for vv in per:
                v = normal_node_num + pre_graph_sum[i] + vv
                if u != v:
                    adjs[0].append(u)
                    adjs[1].append(v)
                    adjs[2].append(1)
        p_adj = sp.csr_matrix((adjs[2], (adjs[0], adjs[1])), shape=(total_num, total_num))
        p_adj = preprocess_adj(p_adj)
        print(f"edge_name={edge_name[i]}, shape={p_adj[2]}, edges_num={len(adjs[0])}")
        whole_support.append(p_adj)

    support = whole_support
    features = sp.csr_matrix(features).tolil()
    features = preprocess_features(features)

    # add hyper nodes information to labels, y_train, y_val, y_test, train_mask, val_mask, test_mask
    hyper_node_labels = [-1 for _ in range(hyper_node_num)]  # using -1 as hyper node's label
    labels = list(labels)
    labels.extend(hyper_node_labels)
    labels = np.array(labels)

    hyper_node_mask = [False for _ in range(hyper_node_num)]
    train_mask = list(train_mask)
    train_mask.extend(hyper_node_mask)
    train_mask = np.array(train_mask, dtype=np.bool)

    val_mask = list(val_mask)
    val_mask.extend(hyper_node_mask)
    val_mask = np.array(val_mask, dtype=np.bool)

    test_mask = list(test_mask)
    test_mask.extend(hyper_node_mask)
    test_mask = np.array(test_mask, dtype=np.bool)

    hyper_node_one_hot = np.zeros((hyper_node_num, args.label_kinds), dtype=np.int32)
    y_train = np.vstack((y_train, hyper_node_one_hot))
    y_val = np.vstack((y_val, hyper_node_one_hot))

    print(f"label_kinds={args.label_kinds} num_supports={num_supports} input_dim={features[2][1]}")
    print(f"total_num = normal_node_num + hyper_node_num = {normal_node_num} + {hyper_node_num} = {total_num}")

    return support, features, y_train, y_val, train_mask, val_mask, hyper_node_num


if __name__ == '__main__':
    from data_utils.data_loader import load_data
    import hparams

    FLAGS = hparams.create()
    vani_adjs, vani_ftr, labels, y_train, y_test, y_val, train_mask, test_mask, val_mask = load_data("data")
    establish(FLAGS, vani_adjs,vani_ftr, labels, y_train, y_test, y_val, train_mask, test_mask, val_mask)
