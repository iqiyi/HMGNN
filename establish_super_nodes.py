# _*_ coding::utf-8 _*_


from sklearn.neighbors import NearestNeighbors
from utils import *


def prepare_super_nodes(args, vani_adj):
    super_nodes = []
    for index, per_graph in enumerate(vani_adj):
        ### 给定一张大图的adj返回这个大图包含多少个小图，每个小图的大小，每个小图包含的id
        def bfs_graph(Graph):
            node_num = Graph[3]
            print("!!!!!!!!!!!!!!")
            print(node_num)
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
        super_nodes.append([graph_num, graph_size, graph_ids])
    return super_nodes


def establish(args, vani_adjs, vani_ftr, labels, y_train, y_test, y_val, train_mask, test_mask, val_mask):
    # graph_info = [normal_node_num, super_node_num, super_links vani_adjs, preprocessed_adjs, preprocessed_feature]
    num_supports = len(vani_adjs)  # 不同种类图的个数
    super_nodes = prepare_super_nodes(args, vani_adjs)  # [[graph_num, graph_size, graph_ids], ...]
    normal_node_num = len(vani_ftr)
    super_node_num = sum([per[0] for per in super_nodes])
    total_num = normal_node_num + super_node_num
    vani_ftr_np = np.array(vani_ftr)

    edge_name = ["e%d" % i for i in range(1, num_supports + 1)]

    print(f"\n normal_nodes_num={normal_node_num}", end=" ")
    for i in range(num_supports):
        print(f"{edge_name[i]}_num={normal_node_num + super_nodes[i][0]}({super_nodes[i][0]})", end=" ")

    # 每张图的节点数量一致，包括三种类型的节点：原始节点、超点、空点

    # 先计算每张图前后图的节点数量
    pre_graph_sum = [0 for _ in range(num_supports)]
    for i in range(1, num_supports):
        pre_graph_sum[i] = pre_graph_sum[i - 1] + super_nodes[i - 1][0]

    features = vani_ftr.copy()
    whole_support = []  # 保存最终每张图预处理后的 邻接矩阵
    for i in range(num_supports):
        # vanilla_adjs --> [[row, col, weight, node_num]]
        for j in range(super_nodes[i][0]):  # 第 j 个超点
            features.append(list(np.mean(vani_ftr_np[super_nodes[i][2][j]], axis=0)))

    # 对于每个超点找 K近邻
    features_np = np.array(features)
    K = 5
    for i in range(num_supports):
        st = normal_node_num + pre_graph_sum[i]
        ed = st + super_nodes[i][0]
        super_features = features_np[st:ed]
        clf = NearestNeighbors(n_neighbors=K + 1, algorithm='ball_tree').fit(super_features)
        distances, indices = clf.kneighbors(super_features)

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
        print(f"{edge_name[i]}, shape = {p_adj[2]} edges_num={len(adjs[0])}")
        whole_support.append(p_adj)

    # print(f"get all whole supports ok, time={time.time() - begin_load_data_time:.3f}s")

    support = whole_support
    features = sp.csr_matrix(features).tolil()
    features = preprocess_features(features)

    # 扩展 labels, y_train, y_val, y_test, train_mask, val_mask, test_mask
    # labels = y_test = np.array(normal_node_num, ), y_train = y_val = np.array(normal_node_num, 2)
    # train_mask = val_mask = test_mask = np.array(noraml_node_num, )
    super_node_labels = [-1 for i in range(super_node_num)]
    labels = list(labels)
    labels.extend(super_node_labels)
    labels = np.array(labels)

    super_node_mask = [False for i in range(super_node_num)]
    train_mask = list(train_mask)
    train_mask.extend(super_node_mask)
    train_mask = np.array(train_mask, dtype=np.bool)

    val_mask = list(val_mask)
    val_mask.extend(super_node_mask)
    val_mask = np.array(val_mask, dtype=np.bool)

    test_mask = list(test_mask)
    test_mask.extend(super_node_mask)
    test_mask = np.array(test_mask, dtype=np.bool)

    super_node_one_hot = np.zeros((super_node_num, 7), dtype=np.int32)
    y_train = np.vstack((y_train, super_node_one_hot))
    y_val = np.vstack((y_val, super_node_one_hot))

    # end_load_data_time = time.time() - begin_load_data_time
    print(f"label_kinds={args.label_kinds} num_supports={num_supports} input_dim={features[2][1]}")
    print(f"total_num = normal_node_num + super_node_num = {normal_node_num} + {super_node_num} = {total_num}")

    return support, features, y_train, y_val, train_mask, val_mask, super_node_num


if __name__ == '__main__':
    from data_utils.data_loader import load_data
    import hparams

    FLAGS = hparams.create()
    vani_adjs, vani_ftr, labels, y_train, y_test, y_val, train_mask, test_mask, val_mask = load_data("data")
    establish(FLAGS, vani_adjs,vani_ftr, labels, y_train, y_test, y_val, train_mask, test_mask, val_mask)
