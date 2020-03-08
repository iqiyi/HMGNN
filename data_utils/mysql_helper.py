# _*_ coding:utf-8 _*_

import pandas as pd
import pymysql
import numpy as np


class MySqlHelper():

    def __init__(self):
        self.host = "zj.columbustabris.w.qiyi.db"
        self.user = "tabris"
        self.passwd = "SHbtyb(z"
        self.db = "columbus_tabris"
        self.port = 1318

        self.con = None
        self.cur = None

    def connect(self):
        self.con = pymysql.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db,
                                   port=self.port)  # connect
        self.cur = self.con.cursor()

    def disconnect(self):
        self.cur.close()
        self.con.close()

    def read_table(self, table_name, del_id=True):
        self.connect()

        sql = "select * from %s" % table_name
        self.cur.execute(sql)
        results = self.cur.fetchall()

        col_result = self.cur.description
        columns = []
        for i in range(len(col_result)):
            columns.append(col_result[i][0])

        df = pd.DataFrame(list(results), columns=columns)
        if del_id and ("id" in columns):
            del df["id"]

        self.disconnect()

        return df

    def read_table_with_date(self,table_name,date_name,start_date,end_date,del_id=False):
        self.connect()

        sql = "select * from %s"%table_name + " where %s between '%s' and '%s'"%(date_name,start_date,end_date)
        self.cur.execute(sql)
        results = self.cur.fetchall()

        col_result = self.cur.description
        columns = []
        for i in range(len(col_result)):
            columns.append(col_result[i][0])

        df = pd.DataFrame(list(results),columns=columns)
        if del_id and ("id" in columns):
            del df["id"]

        self.disconnect()

        return df

    def write_predict_results(self, results, date, table_name):
        self.connect()

        results["date"] = date
        results["pred"] = results["pred"].astype(int)
        results["prob"] = results["prob"].map(lambda x: "%.7f" % x).astype(float)

        uids = results["uid"]
        preds = results["pred"]
        probs = results["prob"]
        dates = results["date"]

        insert_data = list(zip(uids, preds, probs, dates))

        sql = "insert into " + table_name + " (uid,pred,prob,date) values(%s,%s,%s,%s)"
        self.cur.executemany(sql, insert_data)
        self.con.commit()

        self.disconnect()

    def write_predict_gae_results(self, results, date, table_name):
        self.connect()

        results["date"] = date

        uids = results["uid"]
        cluster = results["cluster"]
        dates = results["date"]

        insert_data = list(zip(uids, cluster, dates))

        sql = "insert into " + table_name + " (uid,cluster,date) values(%s,%s,%s)"
        self.cur.executemany(sql, insert_data)
        self.con.commit()

        self.disconnect()
