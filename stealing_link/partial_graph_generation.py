from utils import *
import pickle as pkl
import json
import random
import time
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset', type=str, default='cora', help='citeseer, cora or pubmed')
parser.add_argument(
    '--datapath', type=str, default="data/dataset/original/", help="data path")
parser.add_argument(
    '--prediction_path',
    type=str,
    default='data/pred/',
    help='prediction saved path')
parser.add_argument(
    '--saving_path',
    type=str,
    default='data/partial_graph_with_id/',
    help='partial graph saved path')


args = parser.parse_args()
dataset = args.dataset
datapath = args.datapath
prediction_path = args.prediction_path
saving_path = args.saving_path


def get_link(adj, node_num):
    unlink = []
    link = []
    existing_set = set([])
    rows, cols = adj.nonzero()
    print("There are %d edges in this dataset" % len(rows))
    for i in range(len(rows)):
        r_index = rows[i]
        c_index = cols[i]
        if r_index < c_index:
            link.append([r_index, c_index])
            existing_set.add(",".join([str(r_index), str(c_index)]))

    random.seed(1)
    t_start = time.time()
    while len(unlink) < len(link):
        if len(unlink) % 1000 == 0:
            print(len(unlink), time.time() - t_start)

        row = random.randint(0, node_num - 1)
        col = random.randint(0, node_num - 1)
        if row > col:
            row, col = col, row
        edge_str = ",".join([str(row), str(col)])
        if (row != col) and (edge_str not in existing_set):
            unlink.append([row, col])
            existing_set.add(edge_str)
    return link, unlink


def generate_train_test(link, unlink, dense_pred, gcn_pred, train_ratio):
    train = []
    test = []

    train_len = len(link) * train_ratio
    for i in range(len(link)):
        # print(i)
        link_id0 = link[i][0]
        link_id1 = link[i][1]

        line_link = {
            'label': 1,
            'gcn_pred0': gcn_pred[link_id0],
            'gcn_pred1': gcn_pred[link_id1],
            "dense_pred0": dense_pred[link_id0],
            "dense_pred1": dense_pred[link_id1],
            "feature_arr0": feature_arr[link_id0],
            "feature_arr1": feature_arr[link_id1],
            "id_pair":[int(link_id0),int(link_id1)]
        }

        unlink_id0 = unlink[i][0]
        unlink_id1 = unlink[i][1]

        line_unlink = {
            'label': 0,
            'gcn_pred0': gcn_pred[unlink_id0],
            'gcn_pred1': gcn_pred[unlink_id1],
            "dense_pred0": dense_pred[unlink_id0],
            "dense_pred1": dense_pred[unlink_id1],
            "feature_arr0": feature_arr[unlink_id0],
            "feature_arr1": feature_arr[unlink_id1],
            "id_pair":[int(unlink_id0),int(unlink_id1)]
        }

        if i < train_len:
            train.append(line_link)
            train.append(line_unlink)
        else:
            test.append(line_link)
            test.append(line_unlink)

    with open(
            saving_path + "%s_train_ratio_%0.1f_train.json" %
        (dataset, train_ratio), "w") as wf1, open(
            saving_path + "%s_train_ratio_%0.1f_test.json" %
            (dataset, train_ratio), "w") as wf2:
        for row in train:
            wf1.write("%s\n" % json.dumps(row))
        for row in test:
            wf2.write("%s\n" % json.dumps(row))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(datapath, dataset)
if isinstance(features, np.ndarray):
    feature_arr = features
else:
    feature_arr = features.toarray()
feature_arr = feature_arr.tolist()

dense_pred = pkl.loads(open(prediction_path + "%s_dense_pred.pkl" % dataset, "rb").read())
gcn_pred = pkl.loads(open(prediction_path + "%s_gcn_pred.pkl" % dataset, "rb").read())
dense_pred = dense_pred.tolist()
gcn_pred = gcn_pred.tolist()

node_num = len(dense_pred)
link, unlink = get_link(adj, node_num)
random.shuffle(link)
random.shuffle(unlink)
label = []
for row in link:
    label.append(1)
for row in unlink:
    label.append(0)

# generate 10% to 100% of known edges
t_start = time.time()
for i in range(1, 11):
    print("generating: %d percent" % (i * 10), time.time() - t_start)
    generate_train_test(link, unlink, dense_pred, gcn_pred, i / 10.0)
