'''
Author: Allen
Date: 2021-02-07 12:10:26
LastEditTime: 2021-02-07 13:32:04
LastEditors: Allen
Description: 
FilePath: /stealing_link_publish/stealing_link/attack_0.py
whosyourdaddy
'''
from utils import *
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean
from sklearn.metrics import roc_auc_score
import json
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
    '--partial_graph_path',
    type=str,
    default='data/partial_graph_with_id/',
    help='partial graph saved path')
parser.add_argument('--ratio', type=str, default='0.5', help='(0.1,1.0,0.1)')

args = parser.parse_args()
dataset = args.dataset
datapath = args.datapath
prediction_path = args.prediction_path
partial_graph_path = args.partial_graph_path
ratio = args.ratio


def attack_0(target_posterior_list):
    sim_metric_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    sim_list_target = [[] for _ in range(len(sim_metric_list))]
    for i in range(len(target_posterior_list)):
        for j in range(len(sim_metric_list)):
            # using target only
            target_sim = sim_metric_list[j](target_posterior_list[i][0],
                                            target_posterior_list[i][1])
            sim_list_target[j].append(target_sim)
    return sim_list_target


def write_auc(pred_prob_list, label, desc):
    print("Attack 0 " + desc)
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    with open("result/attack_0.txt", "a") as wf:
        for i in range(len(sim_list_str)):
            pred = np.array(pred_prob_list[i], dtype=np.float64)
            where_are_nan = np.isnan(pred)
            where_are_inf = np.isinf(pred)
            pred[where_are_nan] = 0
            pred[where_are_inf] = 0

            i_auc = roc_auc_score(label, pred)
            if i_auc < 0.5:
                i_auc = 1 - i_auc
            print(sim_list_str[i], i_auc)
            wf.write(
                "%s,%s,%d,%0.5f,%s\n" %
                (dataset, "attack0_%s_%s" %
                 (desc, sim_list_str[i]), -1, i_auc, ratio))


def process():
    # to keep the same testing set for using different ratio of training data,
    # we use 10% of data to evaluate the performance.
    test_path = partial_graph_path + \
        "%s_train_ratio_%s_test.json" % (dataset, "0.9")
    test_data = open(test_path).readlines()  # read test data only
    label_list = []
    target_posterior_list = []
    reference_posterior_list = []
    feature_list = []
    for row in test_data:
        row = json.loads(row)
        label_list.append(row["label"])
        target_posterior_list.append([row["gcn_pred0"], row["gcn_pred1"]])
        reference_posterior_list.append(
            [row["dense_pred0"], row["dense_pred1"]])
        feature_list.append([row["feature_arr0"], row["feature_arr1"]])

    sim_list_target = attack_0(target_posterior_list)
    write_auc(sim_list_target, label_list, desc="target posterior similarity")


if __name__ == "__main__":
    process()
