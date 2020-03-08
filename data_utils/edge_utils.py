# _*_ coding:utf-8 _*_

from mysql_helper import MySqlHelper
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import pickle
import time

flags = tf.app.flags
FLAGS = flags.FLAGS


# tf.app.flags.DEFINE_string('f', '', 'kernel')

class EdgeLoader():
    '''
    2 steps in edge loader:
        1) load edge tables from MySql database
        // 2) select edges in [FLAGS.begin_date, FLAGS.end_date]
        2) merge duplicate edges
    '''

    def __init__(self):
        print(f"\t start loading edges ...")
        begin_time = time.time()

        edge_names_path = FLAGS.preprocessed_data_path + "edge_names.pkl"
        if not FLAGS.use_vanilla_data or not os.path.exists(edge_names_path):
            self._preprocess()

            # save preprocessed data
            if FLAGS.save_vanilla_data:
                if not os.path.exists(FLAGS.preprocessed_data_path):
                    os.makedirs(FLAGS.preprocessed_data_path)
                for index in range(len(self.edges)):
                    cur_path = FLAGS.preprocessed_data_path + 'all_' + self.names[index] + '.csv'
                    print(f"\t\t begin saving data at {cur_path}")
                    self.edges[index].to_csv(cur_path, index=False)  # do not save index

                print(f"\t\t begin saving edge names at {edge_names_path}")
                with open(edge_names_path, 'wb') as f:
                    pickle.dump({'names': self.names}, f)
        else:
            name_path = FLAGS.preprocessed_data_path + 'vanilla_edge_names.pkl'
            print(f"\t\t load edge names from {edge_names_path}")
            with open(edge_names_path, "rb") as f:
                self.names = pickle.load(f)['names']

            self.edges = []
            for index in range(len(self.names)):
                cur_path = FLAGS.preprocessed_data_path + 'all_' + self.names[index] + '.csv'
                print(f"\t\t load edges from {cur_path}")
                self.edges.append(pd.read_csv(cur_path))

        print(f"\t finish loading edges, time elapsed: {time.time() - begin_time:.3f}s ...")

    def _preprocess(self):
        self._load_table()
        #         self._select_edges_by_date()
        self._merge_dup_edges()

    def _load_table(self):
        print(f"\t 1): load edge table from MySql database")
        time_tick = time.time()
        helper = MySqlHelper()
        self.invite_edges = helper.read_table_with_date(FLAGS.invite_edge_table, "dt", FLAGS.begin_date, FLAGS.end_date,
                                                        del_id=True)
        self.device_edges = helper.read_table_with_date(FLAGS.device_edge_table, "dt", FLAGS.begin_date, FLAGS.end_date,
                                                        del_id=True).drop('edge_value', axis=1)
        # self.invite_edges = helper.read_table(FLAGS.invite_edge_table, del_id=True)
        # self.device_edges = helper.read_table(FLAGS.device_edge_table, del_id=True).drop('edge_value', axis=1)

        ip_mask = self.device_edges['type'].apply(lambda x: x == 'ip_ssid')
        did_mask = self.device_edges['type'].apply(lambda x: x == 'did')
        dfp_mask = self.device_edges['type'].apply(lambda x: x == 'dfp')

        self.ip_edges = self.device_edges[ip_mask].drop('type', axis=1)
        self.did_mask = self.device_edges[did_mask].drop('type', axis=1)
        self.dfp_mask = self.device_edges[dfp_mask].drop('type', axis=1)

        self.edges = [self.invite_edges, self.ip_edges, self.did_mask, self.dfp_mask]
        self.names = ['invite', 'ip', 'did', 'dfp']
        print(f"\t time elapsed: {time.time() - time_tick:.3f}s")

    #     def _select_edges_by_date(self):
    #         print(f"\t 2): select edges in [{FLAGS.begin_date}, {FLAGS.end_date}]")
    #         time_tick = time.time()
    #         for index in range(len(self.edges)):
    #             del_mask = self.edges[index]['dt'].apply(lambda x: x < FLAGS.begin_date or x > FLAGS.end_date)
    #             del_row_list = self.edges[index][del_mask].index.tolist()
    #             before_shape = self.edges[index].shape
    #             self.edges[index] = self.edges[index].drop(del_row_list).drop('dt', axis=1)

    #             # add weight column
    #             if 'weight' not in self.edges[index].columns.values:
    #                 self.edges[index]['weight'] = 1
    #             print(f"\t\t edge_name={self.names[index]} before={before_shape} after={self.edges[index].shape}")
    #         print(f"\t time elapsed: {time.time() - time_tick:.3f}s")

    def _merge_dup_edges(self):
        print(f"\t 2): merge duplicate edges")
        time_tick = time.time()

        def weight_sum(df):
            array = df["weight"].values.astype(np.int32)
            return array.sum()

        for index in range(len(self.edges)):
            if 'weight' not in self.edges[index].columns.values:
                self.edges[index]['weight'] = 1
            before_shape = self.edges[index].shape

            # get data at the latest date
            tmp1 = self.edges[index].copy()
            tmp1.drop(columns=["weight"], inplace=True)
            tmp1.sort_values(by="dt", inplace=True)
            tmp1.drop_duplicates(subset=["uid1", "uid2"], keep="last", inplace=True)

            # get the sum of weight for the same uid1 and uid2
            tmp2 = self.edges[index].copy()
            tmp2.drop(columns=["dt"], inplace=True)
            tmp2 = tmp2.groupby(by=['uid1', 'uid2']).apply(
                lambda x: x['weight'].values.astype(np.int32).sum()).reset_index()
            tmp2.rename(columns={0: 'weight'}, inplace=True)

            # merge two dataframes
            self.edges[index] = pd.merge(tmp2, tmp1, on=["uid1", "uid2"], how="inner")
            if self.names[index] != 'invite':
                self.edges[index]['weight'] = 1
            after_shape = self.edges[index].shape

            print(
                f"\t\t {self.names[index]}: before={before_shape} after={after_shape}, time={time.time() - time_tick:.3f}s")
        #             self.edges[index] = self.edges[index].groupby(by=["uid1", "uid2"]).apply(lambda x: weight_sum(x)).reset_index()
        #             self.edges[index].rename(columns={0: 'weight'}, inplace=True)
        #             print(f"\t\t edges from {self.names[index]}: before_shape={before_shape} after_shape={self.edges[index].shape} time={time.time()-time_tick:.3f}s")
        print(f"\t time elapsed: {time.time() - time_tick:.3f}s")

    def get_data(self):
        return self.edges, self.names


if __name__ == "__main__":
    invite_edge_table = 'columbus_common_invite_jisu_material_edges_invite'
    device_edge_table = 'columbus_common_invite_jisu_material_edges_device'
    flags.DEFINE_string('invite_edge_table', invite_edge_table, 'invite edge table name in MySql')
    flags.DEFINE_string('device_edge_table', device_edge_table, 'device edge table name in MySql')

    flags.DEFINE_string('begin_date', '2019-09-15', "begin date")
    flags.DEFINE_string('end_date', '2019-10-17', "end date")
    flags.DEFINE_string('predict_date', '2019-10-11/2019-10-17', 'predict date.')

    flags.DEFINE_boolean('use_vanilla_data', False, 'whether or not use vanilla data')
    flags.DEFINE_boolean('save_vanilla_data', True, 'whether or not save vanilla data')
    flags.DEFINE_boolean('use_preprocessed_data', False, 'whether or not use preprocessed data')
    flags.DEFINE_boolean('save_preprocessed_data', True, 'whether or not save preprocessed data')
    flags.DEFINE_string('preprocessed_data_path', '../preprocessed_data/', 'preprocessed data path')

    loader = EdgeLoader()
    edges, names = loader.get_data()
