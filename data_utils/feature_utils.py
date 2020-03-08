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

class FeatureLoader():
    '''
    4 steps in feature loader:
        1) load feature table from MySql database
        2) drop extram columns
        3) clean duplicate features and only select latest features by date
        4) convert the representation of features into [uid + feature + dt]
    '''

    def __init__(self):
        print(f"\t start loading features ...")
        begin_time = time.time()

        feature_path = FLAGS.preprocessed_data_path + "all_feature.csv"
        if not FLAGS.use_vanilla_data or not os.path.exists(feature_path):
            self._preprocess()

            if FLAGS.save_vanilla_data:
                self.features.to_csv(feature_path, index=False)
        else:
            print(f"\t\t load features from {feature_path}")
            self.features = pd.read_csv(feature_path)
        end_time = time.time() - begin_time
        print(f"\t finish loading features, time elapsed: {end_time:.3f}s ...")

    def _preprocess(self):
        self._load_table()
        self._drop_columns()
        self._clean_dup_uid()
        self._convert()

    def _load_table(self):
        print(f"\t 1): load feature table from MySql database")
        time_tick = time.time()
        helper = MySqlHelper()
        self.features = helper.read_table_with_date(FLAGS.feature_table, "dt", FLAGS.begin_date, FLAGS.end_date,
                                                    del_id=True)
        print(f"\t\t time elapsed: {time.time() - time_tick:.3f}s")

    def _drop_columns(self):
        print(f"\t 2): drop extra columns")
        before_shape = self.features.shape
        drop_columns = ['biz_rule_uid_city_bin_vec', 'biz_rule_uid_province_bin_vec', 'biz_rule_uid_ua_char_freq_vec',
                        'biz_rule_uid_mobile_city_one_hot_vec', 'biz_rule_uid_mobile_province_one_hot_vec',
                        'reg_uid_reg_city_one_hot_vec', 'reg_uid_reg_province_one_hot_vec',
                        'login_uid_city_bin_vec', 'login_uid_province_bin_vec', 'reg_time',
                        'rule_uid_businessName_count', 'rule_uid_record_count', 'rule_uid_city_count',
                        'rule_uid_province_count']  # risk_log create this problem
        '''
        'login_uid_hour_freq_vec', 'biz_rule_invite_uid_hour_freq_vec', 'biz_rule_uid_hour_freq_vec', 'rule_uid_hour_freq_vec',
        '''
        self.features = self.features.drop(drop_columns, axis=1)
        print(f"\t\t delete extra columns, before_shape={before_shape} after_shape={self.features.shape}")

    def _clean_dup_uid(self):
        print(f"\t 3): merge duplicate feature data and only select latest features for each uid")
        time_tick = time.time()
        self.features.sort_values(by=["dt"], ascending=(True), inplace=True)
        self.features.drop_duplicates(subset=["uid"], keep="last", inplace=True)
        print(f"\t\t features_shape={self.features.shape} time={time.time() - time_tick:.3f}s")

    def _convert(self):
        print(f"\t 4): convert the representation of features into [uid + feature + dt]")
        time_tick = time.time()
        column_names = self.features.columns.values.tolist()
        tmp_features = pd.DataFrame(columns=["uid", "feature", "dt"])
        time_tick = time.time()
        first = True
        for index, column_name in enumerate(column_names):
            if column_name in ['uid', 'dt']:
                tmp_features[column_name] = self.features[column_name]
                continue
            tmp = self.features[column_name]
            if column_name.split('_')[-1] != 'vec':
                tmp = tmp.map(lambda x: str(x))

            if first:
                tmp_features['feature'] = tmp
                first = False
            else:
                tmp_features['feature'] = tmp_features['feature'].str.cat(tmp, sep=',')
            if index > 0 and (index % 40 == 0 or index + 1 > len(column_names)):
                print(f"\t\t index = {index} time elapsed = {time.time() - time_tick:.3f}s")

        self.features = tmp_features
        self.features_num = len(self.features["feature"][0].split(","))

        print(
            f"\t\t features.shape={self.features.shape} features_num={self.features_num} time={time.time() - time_tick:.3f}s")

    def get_data(self):
        return self.features


if __name__ == "__main__":
    feature_table = 'columbus_common_invite_jisu_material_features_v1'
    flags.DEFINE_string('begin_date', '2019-09-15', "begin date")
    flags.DEFINE_string('end_date', '2019-10-17', "end date")
    flags.DEFINE_string('predict_date', '2019-10-11/2019-10-17', 'predict date.')
    flags.DEFINE_string('feature_table', feature_table, 'feature table name in MySql')

    flags.DEFINE_boolean('use_vanilla_data', False, 'whether or not use vanilla data')
    flags.DEFINE_boolean('save_vanilla_data', True, 'whether or not save vanilla data')
    flags.DEFINE_boolean('use_preprocessed_data', False, 'whether or not use preprocessed data')
    flags.DEFINE_boolean('save_preprocessed_data', True, 'whether or not save preprocessed data')
    flags.DEFINE_string('preprocessed_data_path', '../preprocessed_data/', 'preprocessed data path')

    loader = FeatureLoader()
    _ = loader.get_data()
