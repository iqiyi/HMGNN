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

class LabelLoader():
    '''
    3 steps in label loader:
        1) load label table from MySql database
        2) drop extra columns and only reserve [uid + label] columns
        3) clean duplicate labels and only select LARGEST and latest labels by date
    '''

    def __init__(self):
        print(f"\t start loading labels ...")
        begin_time = time.time()

        label_path = FLAGS.preprocessed_data_path + "all_label.csv"
        if not FLAGS.use_vanilla_data or not os.path.exists(label_path):
            self._preprocess()

            if FLAGS.save_vanilla_data:
                self.labels.to_csv(label_path, index=False)
        else:
            print(f"\t\t load labels from {label_path}")
            self.labels = pd.read_csv(label_path)

        white_num = sum(self.labels['label'] == 0)
        black_num = sum(self.labels['label'] == 1)
        unlabel_num = sum(self.labels['label'] == -1)
        print(
            f"\t final_labels_shape = {self.labels.shape} white_num={white_num} black_num={black_num} unlabel_num={unlabel_num}")
        print(f"\t finish loading labels, time elapsed: {time.time() - begin_time:.3f}s ...")

    def _preprocess(self):
        self._load_table()
        self._drop_columns()
        self._clean_dup_uid()

    def _load_table(self):
        print(f"\t 1): load label table from MySql database")
        time_tick = time.time()
        helper = MySqlHelper()

        self.labels = helper.read_table_with_date(FLAGS.label_table, "dt", FLAGS.begin_date, FLAGS.end_date,
                                                  del_id=True)

        '''
        self.labels = helper.read_table(FLAGS.label_table, del_id=True)

        # delete features that are not in range [FLAGS.begin_date, FLAGS.end_date]
        del_mask = self.labels['dt'].apply(lambda x: x < FLAGS.begin_date or x > FLAGS.end_date)
        del_row_list = self.labels[del_mask].index.tolist()
        before_shape = self.labels.shape
        self.labels = self.labels.drop(del_row_list)

        print(f"\t\t len(del_row_list)={len(del_row_list)} before={before_shape} after_shape={self.labels.shape} time={time.time()-time_tick:.3f}s")
        '''
        print(
            f"\t\t delete rows whose dates are not in range [begin_date, end_date]=[{FLAGS.begin_date}, {FLAGS.end_date}]")

    def _drop_columns(self):
        print(f"\t 2): drop extra columns and only reserve [uid + label] columns")
        time_tick = time.time()
        before_shape = self.labels.shape
        del_columns = ["phone", "z_black_phone", "updated_at", "black_phone_valid", "vip_level", "fpay",
                       "pay_last_year", \
                       "white_type", "confidence", "black_reason", "level"]
        self.labels = self.labels.drop(del_columns, axis=1)
        print(
            f"\t\t delete extra columns, before={before_shape} after={self.labels.shape} time={time.time() - time_tick:.3f}s")

    def _clean_dup_uid(self):
        print(f"\t 3): merge duplicate label data and only select latest label for each uid")
        time_tick = time.time()
        self.labels.sort_values(by=["label", "dt"], ascending=(True, True), inplace=True)
        self.labels.drop_duplicates(subset=["uid"], keep="last", inplace=True)
        print(f"\t\t labels_shape={self.labels.shape} time={time.time() - time_tick:.3f}s")

    def get_data(self):
        return self.labels


if __name__ == "__main__":
    label_table = 'columbus_common_invite_jisu_material_labels'
    flags.DEFINE_string('label_table', label_table, 'label table name in MySql')

    flags.DEFINE_string('begin_date', '2019-09-15', "begin date")
    flags.DEFINE_string('end_date', '2019-10-17', "end date")
    flags.DEFINE_string('predict_date', '2019-10-11/2019-10-17', 'predict date.')

    flags.DEFINE_boolean('filter_label', False, "whether or not filter labels")

    flags.DEFINE_boolean('use_vanilla_data', False, 'whether or not use vanilla data')
    flags.DEFINE_boolean('save_vanilla_data', True, 'whether or not save vanilla data')
    flags.DEFINE_boolean('use_preprocessed_data', True, 'whether or not use preprocessed data')
    flags.DEFINE_boolean('save_preprocessed_data', True, 'whether or not save preprocessed data')
    flags.DEFINE_string('preprocessed_data_path', '../preprocessed_data/', 'preprocessed data path')

    loader = LabelLoader()
    _ = loader.get_data()
