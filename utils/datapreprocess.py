# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
import lightgbm as lgb
import seaborn as sns
import datetime
import gc
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

class jdata_process():
    def __init__(self):
        self.jdata_action = pd.read_csv("./jdata/jdata_action.csv")  # 用户行为数据
        self.jdata_product = pd.read_csv("./jdata/jdata_product.csv")  # 商品数据
        self.jdata_shop = pd.read_csv("./jdata/jdata_shop.csv")  # 商店数据
        self.jdata_shop = self.jdata_shop.rename(columns={'cate': 'main_cate'})
        self.jdata_user = pd.read_csv("./jdata/jdata_user.csv")  # 用户数据
        self.jdata_comment = pd.read_csv("./jdata/jdata_comment.csv") #用户评论数据
        self.jdata_action = pd.merge(self.jdata_action, self.jdata_product, on="sku_id", how='inner')
        self.jdata_action = pd.merge(self.jdata_action, self.jdata_shop, on='shop_id', how='inner')
        self.jdata_action = pd.merge(self.jdata_action, self.jdata_user, on='user_id', how='inner')
        self.jdata_action['action_date'] = self.jdata_action.action_time.map(lambda x: str(x)[0:10])

        sku_comment_data = self.jdata_comment.groupby('sku_id').sum().reset_index().rename(columns={
            'comments': 'sku_comment_nums',
            'good_comments': 'sku_good_comment_nums',
            'bad_comments': 'sku_bad_comment_nums'
        })
        mean_good_comment_rate = sum(sku_comment_data.sku_good_comment_nums) * 1.0 / sum(
            sku_comment_data.sku_comment_nums)
        mean_bad_comment_rate = sum(sku_comment_data.sku_bad_comment_nums) * 1.0 / sum(
            sku_comment_data.sku_comment_nums)
        sku_comment_data['sku_good_comment_rate'] = (sku_comment_data[
                                                         'sku_good_comment_nums'] + 50 * mean_good_comment_rate) / (
                                                            sku_comment_data['sku_comment_nums'] + 50)
        sku_comment_data['sku_bad_comment_rate'] = (sku_comment_data[
                                                        'sku_bad_comment_nums'] + 50 * mean_bad_comment_rate) / (
                                                           sku_comment_data['sku_comment_nums'] + 50)
        self.jdata_product = pd.merge(self.jdata_product, sku_comment_data, on='sku_id', how='left')

    def get_user_classes(self):
        user_action_times = self.jdata_action.groupby(['user_id']).size().reset_index().rename(columns={0: 'user_action_times'})
        user_action5_times = self.jdata_action[self.jdata_action.type == 5].groupby(['user_id']).size().reset_index().rename(
            columns={0: 'user_action5_times'})
        user_action4_times = self.jdata_action[self.jdata_action.type == 4].groupby(['user_id']).size().reset_index().rename(
            columns={0: 'user_action4_times'})
        user_action3_times = self.jdata_action[self.jdata_action.type == 5].groupby(['user_id']).size().reset_index().rename(
            columns={0: 'user_action3_times'})
        user_action2_times = self.jdata_action[self.jdata_action.type == 2].groupby(['user_id']).size().reset_index().rename(
            columns={0: 'user_action2_times'})
        user_action1_times = self.jdata_action[self.jdata_action.type == 1].groupby(['user_id']).size().reset_index().rename(
            columns={0: 'user_action1_times'})
        user_action_times = pd.merge(user_action_times, user_action1_times, on='user_id', how='outer')
        user_action_times = pd.merge(user_action_times, user_action2_times, on='user_id', how='outer')
        user_action_times = pd.merge(user_action_times, user_action3_times, on='user_id', how='outer')
        user_action_times = pd.merge(user_action_times, user_action4_times, on='user_id', how='outer')
        user_action_times = pd.merge(user_action_times, user_action5_times, on='user_id', how='outer')
        user_action_times = user_action_times.fillna(0)
        del user_action1_times, user_action2_times, user_action3_times, user_action4_times, user_action5_times
        gc.collect()

        # 这里进行用户分群
        # 第一类用户，发现存在只有type==2操作行为的，这部分考虑通过规则来提取，最后汇入结果看收益，当前规则 最后七天存在两次购物行为
        class1_users = user_action_times[user_action_times.user_action2_times == user_action_times.user_action_times].user_id
        # 第二类用户，有过加购物车行为的用户
        class2_users = user_action_times[user_action_times.user_action5_times > 0].user_id
        # 第三类用户，除去前两类用户的结果
        class3_users = user_action_times[(~user_action_times.user_id.isin(class1_users)) & (~user_action_times.user_id.isin(class2_users))].user_id

    # 获得样本 三天内有购买
    def get_samples(self,samples,dat,dat2):
        sample = samples[samples.action_date==dat]
        target = self.jdata_action[(self.jdata_action.type==2) & (self.jdata_action.action_date>dat)& (self.jdata_action.action_date<=dat2)].groupby(['user_id','cate','shop_id']).size().reset_index().rename(columns={0:'flag'})
        sample = pd.merge(sample,target,on=['user_id','cate','shop_id'],how='left')
        sample = sample.fillna(0)
        sample['flag'] = sample.flag.map(lambda x: 1 if x>=1 else 0)
        return sample

    def gen_samples(self):
        samples = self.jdata_action.groupby(['user_id', 'cate', 'shop_id', 'action_date']).size().reset_index()[['user_id', 'cate', 'shop_id', 'action_date']]
        samples0408 = self.get_samples(samples,"2018-04-08", "2018-04-11")
        samples0409 = self.get_samples(samples,"2018-04-09", "2018-04-12")
        samples0410 = self.get_samples(samples,"2018-04-10", "2018-04-13")
        samples0411 = self.get_samples(samples,"2018-04-11", "2018-04-14")
        samples0412 = self.get_samples(samples,"2018-04-12", "2018-04-15")
        submit_sample1 = self.get_samples(samples,"2018-04-15", "2018-04-18")
        submit_sample2 = self.get_samples(samples,"2018-04-14", "2018-04-17")
        submit_sample3 = self.get_samples(samples,"2018-04-13", "2018-04-16")
        return  pd.concat([samples0408, samples0409, samples0410, samples0411, samples0412,submit_sample1,submit_sample2,submit_sample3])

    def add_static_fea(self,sample):
        # 加入静态属性特征
        sample = pd.merge(sample, self.jdata_shop, on='shop_id', how='left')
        sample = pd.merge(sample, self.jdata_user, on='user_id', how='left')
        sample = sample[~sample.shop_reg_tm.isnull()]
        sample['shop_reg_year'] = sample.shop_reg_tm.map(lambda x: int(str(x)[0:4]))
        sample['shop_reg_days'] = sample.shop_reg_tm.map(lambda x: np.NAN if str(x) == 'nan' else (datetime.date(2019, 1, 1) - datetime.date(int(str(x)[0:4]), int(str(x)[5:7]),int(str(x)[8:10]))).days)
        sample['same_with_main_cate'] = (sample.main_cate == sample.cate).map(lambda x: 1 if x else 0)
        sample['user_reg_year'] = sample.user_reg_tm.map(lambda x: int(str(x)[0:4]))
        sample['user_reg_days'] = sample.user_reg_tm.map(lambda x: np.NAN if str(x) == 'nan' else (datetime.date(2019, 1, 1) - datetime.date(int(str(x)[0:4]), int(str(x)[5:7]),int(str(x)[8:10]))).days)
        return sample

    def add_comment_fea(self,sample):
        # shop static feature
        shop_own_features = self.jdata_product.groupby('shop_id')['sku_id', 'brand', 'cate'].nunique().reset_index().rename(
            columns={
                'sku_id': 'shop_own_skus',
                'brand': 'shop_own_brands',
                'cate': 'shop_own_cates'
            })
        shop_comment_fea1 = self.jdata_product.groupby('shop_id')[
            'sku_comment_nums', 'sku_good_comment_nums', 'sku_bad_comment_nums'].sum().reset_index().rename(columns={
            'sku_comment_nums': 'shop_comment_nums',
            'sku_good_comment_nums': 'shop_good_comment_nums',
            'sku_bad_comment_nums': 'shop_bad_comment_nums'
        })
        shop_comment_fea2 = self.jdata_product.groupby('shop_id')['sku_good_comment_rate', 'sku_bad_comment_rate'].agg(
            ['mean', 'max', 'min']).reset_index()
        shop_comment_fea2.columns = shop_comment_fea2.columns.droplevel(0)
        shop_comment_fea2.columns = ['shop_id', 'shop_good_comm_rate_mean', 'shop_good_comm_rate_max',
                                     'shop_good_comm_rate_min', \
                                     'shop_bad_comm_rate_mean', 'shop_bad_comm_rate_max', 'shop_bad_comm_rate_min']

        # cate static feature
        cate_own_features = self.jdata_product.groupby('cate')['sku_id', 'brand', 'shop_id'].nunique().reset_index().rename(
            columns={
                'sku_id': 'cate_own_skus',
                'brand': 'cate_own_brands',
                'shop_id': 'cate_own_shops'
            })
        cate_comment_fea1 = self.jdata_product.groupby('cate')[
            'sku_comment_nums', 'sku_good_comment_nums', 'sku_bad_comment_nums'].sum().reset_index().rename(columns={
            'sku_comment_nums': 'cate_comment_nums',
            'sku_good_comment_nums': 'cate_good_comment_nums',
            'sku_bad_comment_nums': 'cate_bad_comment_nums'
        })
        cate_comment_fea2 = self.jdata_product.groupby('cate')['sku_good_comment_rate', 'sku_bad_comment_rate'].agg(
            ['mean', 'max', 'min']).reset_index()
        cate_comment_fea2.columns = cate_comment_fea2.columns.droplevel(0)
        cate_comment_fea2.columns = ['cate', 'cate_good_comm_rate_mean', 'cate_good_comm_rate_max',
                                     'cate_good_comm_rate_min', \
                                     'cate_bad_comm_rate_mean', 'cate_bad_comm_rate_max', 'cate_bad_comm_rate_min']
        # shop and cate  And Sort Features like which is the most popular cate in a shop
        shop_cate_own_features = self.jdata_product.groupby(['shop_id', 'cate'])[
            'sku_id', 'brand'].nunique().reset_index().rename(columns={
            'sku_id': 'shop_cate_own_skus',
            'brand': 'shop_cate_own_brands'
        })
        shop_cate_comment_fea1 = self.jdata_product.groupby(['shop_id', 'cate'])[
            'sku_comment_nums', 'sku_good_comment_nums', 'sku_bad_comment_nums'].sum().reset_index().rename(columns={
            'sku_comment_nums': 'shop_cate_comment_nums',
            'sku_good_comment_nums': 'shop_cate_good_comment_nums',
            'sku_bad_comment_nums': 'shop_cate_bad_comment_nums'
        })
        shop_cate_comment_fea2 = self.jdata_product.groupby(['shop_id', 'cate'])[
            'sku_good_comment_rate', 'sku_bad_comment_rate'].agg(['mean', 'max', 'min']).reset_index()
        shop_cate_comment_fea2.columns = shop_cate_comment_fea2.columns.droplevel(0)
        shop_cate_comment_fea2.columns = ['shop_id', 'cate', 'shop_cate_good_comm_rate_mean',
                                          'shop_cate_good_comm_rate_max', 'shop_cate_good_comm_rate_min', \
                                          'shop_cate_bad_comm_rate_mean', 'shop_cate_bad_comm_rate_max',
                                          'shop_cate_bad_comm_rate_min']

        shop_static_fea = pd.merge(shop_own_features, shop_comment_fea1, on='shop_id', how='outer')
        shop_static_fea = pd.merge(shop_static_fea, shop_comment_fea2, on='shop_id', how='outer')
        shop_static_fea = shop_static_fea.fillna(shop_static_fea.mean())
        cate_static_fea = pd.merge(cate_own_features, cate_comment_fea1, on='cate', how='outer')
        cate_static_fea = pd.merge(cate_static_fea, cate_comment_fea2, on='cate', how='outer')
        cate_static_fea = cate_static_fea.fillna(cate_static_fea.mean())
        shop_cate_static_fea = pd.merge(shop_cate_own_features, shop_cate_comment_fea1, on=['shop_id', 'cate'], how='outer')
        shop_cate_static_fea = pd.merge(shop_cate_static_fea, shop_cate_comment_fea2, on=['shop_id', 'cate'], how='outer')
        shop_cate_static_fea = shop_cate_static_fea.fillna(shop_cate_static_fea.mean())

        sample = pd.merge(sample, shop_static_fea, on='shop_id')
        sample = pd.merge(sample, cate_static_fea, on='cate')
        sample = pd.merge(sample, shop_cate_static_fea, on=['shop_id', 'cate'])
        return sample

    # 获得历史上的转化率数据
    def get_Conversion1_rate(self, date, cols, change_type):
        tmp = self.jdata_action[self.jdata_action.action_date < date]
        mean_rate = sum(tmp.type == change_type) * 1.0 / sum(tmp.type == 1)
        data = pd.merge(
            tmp[tmp.type == 1].groupby(cols).size().reset_index().rename(columns={0: cols + '_action1_counts'}), \
            tmp[tmp.type == change_type].groupby(cols).size().reset_index().rename(
                columns={0: cols + '_action2_counts'}), on=cols, how='left')

        data = data.fillna(0)
        data[cols + '_action1_counts'] = data[cols + '_action1_counts'].map(lambda x: x + 1000)
        data[cols + '_action2_counts'] = data[cols + '_action2_counts'].map(lambda x: x + 1000 * mean_rate)
        data[cols + '_' + str(change_type) + '_coversion_rate'] = data[cols + '_action2_counts'] * 1.0 / data[
            cols + '_action1_counts']
        return data[[cols, cols + '_' + str(change_type) + '_coversion_rate']]

    # 获得历史上的转化率合并数据
    def get_Conversion2_rate(self,date, cols1, cols2, change_type):
        tmp = self.jdata_action[self.jdata_action.action_date < date]
        mean_rate = sum(tmp.type == change_type) * 1.0 / sum(tmp.type == 1)
        data = pd.merge(tmp[tmp.type == 1].groupby([cols1, cols2]).size().reset_index().rename(
            columns={0: cols1 + "_" + cols2 + '_action1_counts'}), \
                        tmp[tmp.type == change_type].groupby([cols1, cols2]).size().reset_index().rename(
                            columns={0: cols1 + "_" + cols2 + '_action2_counts'}), on=[cols1, cols2], how='left')

        data = data.fillna(0)
        data[cols1 + "_" + cols2 + '_action1_counts'] = data[cols1 + "_" + cols2 + '_action1_counts'].map(lambda x: x + 500)
        data[cols1 + "_" + cols2 + '_action2_counts'] = data[cols1 + "_" + cols2 + '_action2_counts'].map(lambda x: x + 500 * mean_rate)
        data[cols1 + "_" + cols2 + '_' + str(change_type) + '_coversion_rate'] = data[cols1 + "_" + cols2 + '_action2_counts'] * 1.0 / \
                                                                                    data[cols1 + "_" + cols2 + '_action1_counts']
        return data[[cols1, cols2, cols1 + "_" + cols2 + '_' + str(change_type) + '_coversion_rate']]

    def add_conversion1_fea(self, sample):
        cate_2_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'cate', 2)
        cate_3_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'cate', 3)
        cate_4_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'cate', 4)
        cate_conversion_rate = pd.merge(cate_2_Conversion_rate, cate_3_Conversion_rate, on='cate', how='outer')
        cate_conversion_rate = pd.merge(cate_conversion_rate, cate_4_Conversion_rate, on='cate', how='outer')
        user_id_2_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'user_id', 2)
        user_id_3_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'user_id', 3)
        user_id_4_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'user_id', 4)
        user_id_conversion_rate = pd.merge(user_id_2_Conversion_rate, user_id_3_Conversion_rate, on='user_id', how='outer')
        user_id_conversion_rate = pd.merge(user_id_conversion_rate, user_id_4_Conversion_rate, on='user_id', how='outer')
        shop_2_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'shop_id', 2)
        shop_3_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'shop_id', 3)
        shop_4_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'shop_id', 4)
        shop_conversion_rate = pd.merge(shop_2_Conversion_rate, shop_3_Conversion_rate, on='shop_id', how='outer')
        shop_conversion_rate = pd.merge(shop_conversion_rate, shop_4_Conversion_rate, on='shop_id', how='outer')

        province_2_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'province', 2)
        province_3_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'province', 3)
        province_4_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'province', 4)
        province_conversion_rate = pd.merge(province_2_Conversion_rate, province_3_Conversion_rate, on='province', how='outer')
        province_conversion_rate = pd.merge(province_conversion_rate, province_4_Conversion_rate, on='province', how='outer')

        city_2_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'city', 2)
        city_3_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'city', 3)
        city_4_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'city', 4)
        city_conversion_rate = pd.merge(city_2_Conversion_rate, city_3_Conversion_rate, on='city', how='outer')
        city_conversion_rate = pd.merge(city_conversion_rate, city_4_Conversion_rate, on='city', how='outer')

        county_2_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'county', 2)
        county_3_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'county', 3)
        county_4_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'county', 4)
        county_conversion_rate = pd.merge(county_2_Conversion_rate, county_3_Conversion_rate, on='county', how='outer')
        county_conversion_rate = pd.merge(county_conversion_rate, county_4_Conversion_rate, on='county', how='outer')

        age_2_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'age', 2)
        age_3_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'age', 3)
        age_4_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'age', 4)
        age_conversion_rate = pd.merge(age_2_Conversion_rate, age_3_Conversion_rate, on='age', how='outer')
        age_conversion_rate = pd.merge(age_conversion_rate, age_4_Conversion_rate, on='age', how='outer')

        sex_2_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'sex', 2)
        sex_3_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'sex', 3)
        sex_4_Conversion_rate = self.get_Conversion1_rate("2018-04-08", 'sex', 4)
        sex_conversion_rate = pd.merge(sex_2_Conversion_rate, sex_3_Conversion_rate, on='sex', how='outer')
        sex_conversion_rate = pd.merge(sex_conversion_rate, sex_4_Conversion_rate, on='sex', how='outer')

        sample = pd.merge(sample, cate_conversion_rate, on='cate', how='left')
        sample = pd.merge(sample, user_id_conversion_rate, on='user_id', how='left')
        sample = pd.merge(sample, shop_conversion_rate, on='shop_id', how='left')
        sample = pd.merge(sample, province_conversion_rate, on='province', how='left')
        sample = pd.merge(sample, city_conversion_rate, on='city', how='left')
        sample = pd.merge(sample, county_conversion_rate, on='county', how='left')
        sample = pd.merge(sample, age_conversion_rate, on='age', how='left')
        sample = pd.merge(sample, sex_conversion_rate, on='sex', how='left')
        return sample

    def add_conversion2_fea(self, sample):
        # user_cate
        user_cate_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'user_id', 'cate', 2)
        user_cate_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'user_id', 'cate', 3)
        user_cate_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'user_id', 'cate', 4)
        user_cate_conversion_rate = pd.merge(user_cate_2_Conversion_rate, user_cate_3_Conversion_rate,
                                             on=['user_id', 'cate'], how='outer')
        user_cate_conversion_rate = pd.merge(user_cate_conversion_rate, user_cate_4_Conversion_rate,
                                             on=['user_id', 'cate'], how='outer')
        del user_cate_2_Conversion_rate, user_cate_3_Conversion_rate, user_cate_4_Conversion_rate

        # user_shop
        user_shop_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'user_id', 'shop_id', 2)
        user_shop_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'user_id', 'shop_id', 3)
        user_shop_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'user_id', 'shop_id', 4)
        user_shop_conversion_rate = pd.merge(user_shop_2_Conversion_rate, user_shop_3_Conversion_rate,
                                             on=['user_id', 'shop_id'], how='outer')
        user_shop_conversion_rate = pd.merge(user_shop_conversion_rate, user_shop_4_Conversion_rate,
                                             on=['user_id', 'shop_id'], how='outer')
        del user_shop_2_Conversion_rate, user_shop_3_Conversion_rate, user_shop_4_Conversion_rate

        # cate_shop
        cate_shop_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'cate', 'shop_id', 2)
        cate_shop_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'cate', 'shop_id', 3)
        cate_shop_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'cate', 'shop_id', 4)
        cate_shop_conversion_rate = pd.merge(cate_shop_2_Conversion_rate, cate_shop_3_Conversion_rate,
                                             on=['cate', 'shop_id'], how='outer')
        cate_shop_conversion_rate = pd.merge(cate_shop_conversion_rate, cate_shop_4_Conversion_rate,
                                             on=['cate', 'shop_id'], how='outer')
        del cate_shop_2_Conversion_rate, cate_shop_3_Conversion_rate, cate_shop_4_Conversion_rate

        # age_shop
        age_shop_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'shop_id', 2)
        age_shop_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'shop_id', 3)
        age_shop_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'shop_id', 4)
        age_shop_conversion_rate = pd.merge(age_shop_2_Conversion_rate, age_shop_3_Conversion_rate,
                                            on=['age', 'shop_id'], how='outer')
        age_shop_conversion_rate = pd.merge(age_shop_conversion_rate, age_shop_4_Conversion_rate, on=['age', 'shop_id'],
                                            how='outer')
        del age_shop_2_Conversion_rate, age_shop_3_Conversion_rate, age_shop_4_Conversion_rate

        # age_cate
        age_cate_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'cate', 2)
        age_cate_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'cate', 3)
        age_cate_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'cate', 4)
        age_cate_conversion_rate = pd.merge(age_cate_2_Conversion_rate, age_cate_3_Conversion_rate, on=['age', 'cate'],
                                            how='outer')
        age_cate_conversion_rate = pd.merge(age_cate_conversion_rate, age_cate_4_Conversion_rate, on=['age', 'cate'],
                                            how='outer')
        del age_cate_2_Conversion_rate, age_cate_3_Conversion_rate, age_cate_4_Conversion_rate

        # age_province
        age_province_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'province', 2)
        age_province_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'province', 3)
        age_province_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'province', 4)
        age_province_conversion_rate = pd.merge(age_province_2_Conversion_rate, age_province_3_Conversion_rate,
                                                on=['age', 'province'], how='outer')
        age_province_conversion_rate = pd.merge(age_province_conversion_rate, age_province_4_Conversion_rate,
                                                on=['age', 'province'], how='outer')
        del age_province_2_Conversion_rate, age_province_3_Conversion_rate, age_province_4_Conversion_rate

        age_city_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'city', 2)
        age_city_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'city', 3)
        age_city_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'city', 4)
        age_city_conversion_rate = pd.merge(age_city_2_Conversion_rate, age_city_3_Conversion_rate, on=['age', 'city'],
                                            how='outer')
        age_city_conversion_rate = pd.merge(age_city_conversion_rate, age_city_4_Conversion_rate, on=['age', 'city'],
                                            how='outer')
        del age_city_2_Conversion_rate, age_city_3_Conversion_rate, age_city_4_Conversion_rate

        age_county_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'county', 2)
        age_county_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'county', 3)
        age_county_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'age', 'county', 4)
        age_county_conversion_rate = pd.merge(age_county_2_Conversion_rate, age_county_3_Conversion_rate,
                                              on=['age', 'county'], how='outer')
        age_county_conversion_rate = pd.merge(age_county_conversion_rate, age_county_4_Conversion_rate,
                                              on=['age', 'county'], how='outer')

        # sex_shop
        sex_shop_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'shop_id', 2)
        sex_shop_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'shop_id', 3)
        sex_shop_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'shop_id', 4)
        sex_shop_conversion_rate = pd.merge(sex_shop_2_Conversion_rate, sex_shop_3_Conversion_rate,
                                            on=['sex', 'shop_id'], how='outer')
        sex_shop_conversion_rate = pd.merge(sex_shop_conversion_rate, sex_shop_4_Conversion_rate, on=['sex', 'shop_id'],
                                            how='outer')

        # sex_cate
        sex_cate_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'cate', 2)
        sex_cate_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'cate', 3)
        sex_cate_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'cate', 4)
        sex_cate_conversion_rate = pd.merge(sex_cate_2_Conversion_rate, sex_cate_3_Conversion_rate, on=['sex', 'cate'],
                                            how='outer')
        sex_cate_conversion_rate = pd.merge(sex_cate_conversion_rate, sex_cate_4_Conversion_rate, on=['sex', 'cate'],
                                            how='outer')

        # sex_province
        sex_province_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'province', 2)
        sex_province_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'province', 3)
        sex_province_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'province', 4)
        sex_province_conversion_rate = pd.merge(sex_province_2_Conversion_rate, sex_province_3_Conversion_rate,
                                                on=['sex', 'province'], how='outer')
        sex_province_conversion_rate = pd.merge(sex_province_conversion_rate, sex_province_4_Conversion_rate,
                                                on=['sex', 'province'], how='outer')

        sex_city_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'city', 2)
        sex_city_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'city', 3)
        sex_city_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'city', 4)
        sex_city_conversion_rate = pd.merge(sex_city_2_Conversion_rate, sex_city_3_Conversion_rate, on=['sex', 'city'],
                                            how='outer')
        sex_city_conversion_rate = pd.merge(sex_city_conversion_rate, sex_city_4_Conversion_rate, on=['sex', 'city'],
                                            how='outer')

        sex_county_2_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'county', 2)
        sex_county_3_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'county', 3)
        sex_county_4_Conversion_rate = self.get_Conversion2_rate("2018-04-08", 'sex', 'county', 4)
        sex_county_conversion_rate = pd.merge(sex_county_2_Conversion_rate, sex_county_3_Conversion_rate,
                                              on=['sex', 'county'], how='outer')
        sex_county_conversion_rate = pd.merge(sex_county_conversion_rate, sex_county_4_Conversion_rate,
                                              on=['sex', 'county'], how='outer')

        sample = pd.merge(sample, user_cate_conversion_rate, on=['user_id', 'cate'], how='left')
        sample = pd.merge(sample, user_shop_conversion_rate, on=['user_id', 'shop_id'], how='left')
        sample = pd.merge(sample, cate_shop_conversion_rate, on=['cate', 'shop_id'], how='left')
        sample = pd.merge(sample, age_shop_conversion_rate, on=['age', 'shop_id'], how='left')
        sample = pd.merge(sample, age_cate_conversion_rate, on=['age', 'cate'], how='left')
        sample = pd.merge(sample, age_province_conversion_rate, on=['age', 'province'], how='left')
        sample = pd.merge(sample, age_city_conversion_rate, on=['age', 'city'], how='left')
        sample = pd.merge(sample, age_county_conversion_rate, on=['age', 'county'], how='left')
        sample = pd.merge(sample, sex_shop_conversion_rate, on=['sex', 'shop_id'], how='left')
        sample = pd.merge(sample, sex_cate_conversion_rate, on=['sex', 'cate'], how='left')
        sample = pd.merge(sample, sex_province_conversion_rate, on=['sex', 'province'], how='left')
        sample = pd.merge(sample, sex_city_conversion_rate, on=['sex', 'city'], how='left')
        sample = pd.merge(sample, sex_county_conversion_rate, on=['sex', 'county'], how='left')
        return sample

    def get_user_behavier_fea(self, dat, before_dat, suffix):
        # 商品粒度
        tmp = self.jdata_action[(self.jdata_action.action_date >= before_dat) & (self.jdata_action.action_date <= dat)]

        # 1. 该用户之前时间段浏览天数，浏览的店铺数，购买数，
        sku_action2_counts = tmp[(tmp.type == 2)].groupby('sku_id').size().reset_index().rename(
            columns={0: "sku_action2_counts_" + suffix})
        sku_action3_counts = tmp[(tmp.type == 3)].groupby('sku_id').size().reset_index().rename(
            columns={0: "sku_action3_counts_" + suffix})
        sku_action4_counts = tmp[(tmp.type == 4)].groupby('sku_id').size().reset_index().rename(
            columns={0: "sku_action4_counts_" + suffix})
        sku_action5_counts = tmp[(tmp.type == 5)].groupby('sku_id').size().reset_index().rename(
            columns={0: "sku_action5_counts_" + suffix})
        sku_action1_counts = tmp[(tmp.type == 1)].groupby('sku_id').size().reset_index().rename(
            columns={0: "sku_action1_counts_" + suffix})
        # 用户粒度，该时间段次数
        user_action2_counts = tmp[(tmp.type == 2)].groupby('user_id').size().reset_index().rename(
            columns={0: "user_action2_counts_" + suffix})
        user_action3_counts = tmp[(tmp.type == 3)].groupby('user_id').size().reset_index().rename(
            columns={0: "user_action3_counts_" + suffix})
        user_action4_counts = tmp[(tmp.type == 4)].groupby('user_id').size().reset_index().rename(
            columns={0: "user_action4_counts_" + suffix})
        user_action5_counts = tmp[(tmp.type == 5)].groupby('user_id').size().reset_index().rename(
            columns={0: "user_action5_counts_" + suffix})
        user_action1_counts = tmp[(tmp.type == 1)].groupby('user_id').size().reset_index().rename(
            columns={0: "user_action1_counts_" + suffix})

        user_action2_counts_date = tmp[(tmp.type == 2)].groupby('user_id')[
            'action_date'].nunique().reset_index().rename(columns={'action_date': "user_action2_counts_date_" + suffix})
        user_action3_counts_date = tmp[(tmp.type == 3)].groupby('user_id')[
            'action_date'].nunique().reset_index().rename(columns={'action_date': "user_action3_counts_date_" + suffix})
        user_action4_counts_date = tmp[(tmp.type == 4)].groupby('user_id')[
            'action_date'].nunique().reset_index().rename(columns={'action_date': "user_action4_counts_date_" + suffix})

        # 品类粒度，当天次数
        cate_action2_counts = tmp[(tmp.type == 2)].groupby('cate').size().reset_index().rename(
            columns={0: "cate_action2_counts_" + suffix})
        cate_action3_counts = tmp[(tmp.type == 3)].groupby('cate').size().reset_index().rename(
            columns={0: "cate_action3_counts_" + suffix})
        cate_action4_counts = tmp[(tmp.type == 4)].groupby('cate').size().reset_index().rename(
            columns={0: "cate_action4_counts_" + suffix})
        cate_action5_counts = tmp[(tmp.type == 5)].groupby('cate').size().reset_index().rename(
            columns={0: "cate_action5_counts_" + suffix})
        cate_action1_counts = tmp[(tmp.type == 1)].groupby('cate').size().reset_index().rename(
            columns={0: "cate_action1_counts_" + suffix})
        # shop_id 粒度
        shop_action2_counts = tmp[(tmp.type == 2)].groupby('shop_id').size().reset_index().rename(
            columns={0: "shop_action2_counts_" + suffix})
        shop_action3_counts = tmp[(tmp.type == 3)].groupby('shop_id').size().reset_index().rename(
            columns={0: "shop_action3_counts_" + suffix})
        shop_action4_counts = tmp[(tmp.type == 4)].groupby('shop_id').size().reset_index().rename(
            columns={0: "shop_action4_counts_" + suffix})
        shop_action5_counts = tmp[(tmp.type == 5)].groupby('shop_id').size().reset_index().rename(
            columns={0: "shop_action5_counts_" + suffix})
        shop_action1_counts = tmp[(tmp.type == 1)].groupby('shop_id').size().reset_index().rename(
            columns={0: "shop_action1_counts_" + suffix})
        # province
        province_action2_counts = tmp[(tmp.type == 2)].groupby('province').size().reset_index().rename(
            columns={0: "province_action2_counts_" + suffix})
        province_action3_counts = tmp[(tmp.type == 3)].groupby('province').size().reset_index().rename(
            columns={0: "province_action3_counts_" + suffix})
        province_action4_counts = tmp[(tmp.type == 4)].groupby('province').size().reset_index().rename(
            columns={0: "province_action4_counts_" + suffix})
        province_action5_counts = tmp[(tmp.type == 5)].groupby('province').size().reset_index().rename(
            columns={0: "province_action5_counts_" + suffix})
        province_action1_counts = tmp[(tmp.type == 1)].groupby('province').size().reset_index().rename(
            columns={0: "province_action1_counts_" + suffix})
        # city
        city_action2_counts = tmp[(tmp.type == 2)].groupby('city').size().reset_index().rename(
            columns={0: "city_action2_counts_" + suffix})
        city_action3_counts = tmp[(tmp.type == 3)].groupby('city').size().reset_index().rename(
            columns={0: "city_action3_counts_" + suffix})
        city_action4_counts = tmp[(tmp.type == 4)].groupby('city').size().reset_index().rename(
            columns={0: "city_action4_counts_" + suffix})
        city_action5_counts = tmp[(tmp.type == 5)].groupby('city').size().reset_index().rename(
            columns={0: "city_action5_counts_" + suffix})
        city_action1_counts = tmp[(tmp.type == 1)].groupby('city').size().reset_index().rename(
            columns={0: "city_action1_counts_" + suffix})
        # county
        county_action2_counts = tmp[(tmp.type == 2)].groupby('county').size().reset_index().rename(
            columns={0: "county_action2_counts_" + suffix})
        county_action3_counts = tmp[(tmp.type == 3)].groupby('county').size().reset_index().rename(
            columns={0: "county_action3_counts_" + suffix})
        county_action4_counts = tmp[(tmp.type == 4)].groupby('county').size().reset_index().rename(
            columns={0: "county_action4_counts_" + suffix})
        county_action5_counts = tmp[(tmp.type == 5)].groupby('county').size().reset_index().rename(
            columns={0: "county_action5_counts_" + suffix})
        county_action1_counts = tmp[(tmp.type == 1)].groupby('county').size().reset_index().rename(
            columns={0: "county_action1_counts_" + suffix})
        # brand
        brand_action2_counts = tmp[(tmp.type == 2)].groupby('brand').size().reset_index().rename(
            columns={0: "brand_action2_counts_" + suffix})
        brand_action3_counts = tmp[(tmp.type == 3)].groupby('brand').size().reset_index().rename(
            columns={0: "brand_action3_counts_" + suffix})
        brand_action4_counts = tmp[(tmp.type == 4)].groupby('brand').size().reset_index().rename(
            columns={0: "brand_action4_counts_" + suffix})
        brand_action5_counts = tmp[(tmp.type == 5)].groupby('brand').size().reset_index().rename(
            columns={0: "brand_action5_counts_" + suffix})
        brand_action1_counts = tmp[(tmp.type == 1)].groupby('brand').size().reset_index().rename(
            columns={0: "brand_action1_counts_" + suffix})
        # age
        age_action2_counts = tmp[(tmp.type == 2)].groupby('age').size().reset_index().rename(
            columns={0: "age_action2_counts_" + suffix})
        age_action3_counts = tmp[(tmp.type == 3)].groupby('age').size().reset_index().rename(
            columns={0: "age_action3_counts_" + suffix})
        age_action4_counts = tmp[(tmp.type == 4)].groupby('age').size().reset_index().rename(
            columns={0: "age_action4_counts_" + suffix})
        age_action5_counts = tmp[(tmp.type == 5)].groupby('age').size().reset_index().rename(
            columns={0: "age_action5_counts_" + suffix})
        age_action1_counts = tmp[(tmp.type == 1)].groupby('age').size().reset_index().rename(
            columns={0: "age_action1_counts_" + suffix})
        # sex
        sex_action2_counts = tmp[(tmp.type == 2)].groupby('sex').size().reset_index().rename(
            columns={0: "sex_action2_counts_" + suffix})
        sex_action3_counts = tmp[(tmp.type == 3)].groupby('sex').size().reset_index().rename(
            columns={0: "sex_action3_counts_" + suffix})
        sex_action4_counts = tmp[(tmp.type == 4)].groupby('sex').size().reset_index().rename(
            columns={0: "sex_action4_counts_" + suffix})
        sex_action5_counts = tmp[(tmp.type == 5)].groupby('sex').size().reset_index().rename(
            columns={0: "sex_action5_counts_" + suffix})
        sex_action1_counts = tmp[(tmp.type == 1)].groupby('sex').size().reset_index().rename(
            columns={0: "sex_action1_counts_" + suffix})

        user_sku_action2_counts = tmp[(tmp.type == 2)].groupby(['user_id', 'sku_id']).size().reset_index().rename(
            columns={0: "user_sku_action2_counts_" + suffix})
        user_sku_action3_counts = tmp[(tmp.type == 3)].groupby(['user_id', 'sku_id']).size().reset_index().rename(
            columns={0: "user_sku_action3_counts_" + suffix})
        user_sku_action4_counts = tmp[(tmp.type == 4)].groupby(['user_id', 'sku_id']).size().reset_index().rename(
            columns={0: "user_sku_action4_counts_" + suffix})
        user_sku_action5_counts = tmp[(tmp.type == 5)].groupby(['user_id', 'sku_id']).size().reset_index().rename(
            columns={0: "user_sku_action5_counts_" + suffix})
        user_sku_action1_counts = tmp[(tmp.type == 1)].groupby(['user_id', 'sku_id']).size().reset_index().rename(
            columns={0: "user_sku_action1_counts_" + suffix})
        user_sku_action = pd.merge(user_sku_action1_counts, user_sku_action2_counts, on=['user_id', 'sku_id'],
                                   how='outer')
        user_sku_action = pd.merge(user_sku_action, user_sku_action3_counts, on=['user_id', 'sku_id'], how='outer')
        user_sku_action = pd.merge(user_sku_action, user_sku_action4_counts, on=['user_id', 'sku_id'], how='outer')
        user_sku_action = pd.merge(user_sku_action, user_sku_action5_counts, on=['user_id', 'sku_id'], how='outer')
        user_cate_action2_counts = tmp[(tmp.type == 2)].groupby(['user_id', 'cate']).size().reset_index().rename(
            columns={0: "user_cate_action2_counts_" + suffix})
        user_cate_action3_counts = tmp[(tmp.type == 3)].groupby(['user_id', 'cate']).size().reset_index().rename(
            columns={0: "user_cate_action3_counts_" + suffix})
        user_cate_action4_counts = tmp[(tmp.type == 4)].groupby(['user_id', 'cate']).size().reset_index().rename(
            columns={0: "user_cate_action4_counts_" + suffix})
        user_cate_action5_counts = tmp[(tmp.type == 5)].groupby(['user_id', 'cate']).size().reset_index().rename(
            columns={0: "user_cate_action5_counts_" + suffix})
        user_cate_action1_counts = tmp[(tmp.type == 1)].groupby(['user_id', 'cate']).size().reset_index().rename(
            columns={0: "user_cate_action1_counts_" + suffix})
        user_cate_action = pd.merge(user_cate_action1_counts, user_cate_action2_counts, on=['user_id', 'cate'],
                                    how='outer')
        user_cate_action = pd.merge(user_cate_action, user_cate_action3_counts, on=['user_id', 'cate'], how='outer')
        user_cate_action = pd.merge(user_cate_action, user_cate_action4_counts, on=['user_id', 'cate'], how='outer')
        user_cate_action = pd.merge(user_cate_action, user_cate_action5_counts, on=['user_id', 'cate'], how='outer')
        user_brand_action2_counts = tmp[(tmp.type == 2)].groupby(['user_id', 'brand']).size().reset_index().rename(
            columns={0: "user_brand_action2_counts_" + suffix})
        user_brand_action3_counts = tmp[(tmp.type == 3)].groupby(['user_id', 'brand']).size().reset_index().rename(
            columns={0: "user_brand_action3_counts_" + suffix})
        user_brand_action4_counts = tmp[(tmp.type == 4)].groupby(['user_id', 'brand']).size().reset_index().rename(
            columns={0: "user_brand_action4_counts_" + suffix})
        user_brand_action5_counts = tmp[(tmp.type == 5)].groupby(['user_id', 'brand']).size().reset_index().rename(
            columns={0: "user_brand_action5_counts_" + suffix})
        user_brand_action1_counts = tmp[(tmp.type == 1)].groupby(['user_id', 'brand']).size().reset_index().rename(
            columns={0: "user_brand_action1_counts_" + suffix})
        user_brand_action = pd.merge(user_brand_action1_counts, user_brand_action2_counts, on=['user_id', 'brand'],
                                     how='outer')
        user_brand_action = pd.merge(user_brand_action, user_brand_action3_counts, on=['user_id', 'brand'], how='outer')
        user_brand_action = pd.merge(user_brand_action, user_brand_action4_counts, on=['user_id', 'brand'], how='outer')
        user_brand_action = pd.merge(user_brand_action, user_brand_action5_counts, on=['user_id', 'brand'], how='outer')
        user_shop_action2_counts = tmp[(tmp.type == 2)].groupby(['user_id', 'shop_id']).size().reset_index().rename(
            columns={0: "user_shop_action2_counts_" + suffix})
        user_shop_action3_counts = tmp[(tmp.type == 3)].groupby(['user_id', 'shop_id']).size().reset_index().rename(
            columns={0: "user_shop_action3_counts_" + suffix})
        user_shop_action4_counts = tmp[(tmp.type == 4)].groupby(['user_id', 'shop_id']).size().reset_index().rename(
            columns={0: "user_shop_action4_counts_" + suffix})
        user_shop_action5_counts = tmp[(tmp.type == 5)].groupby(['user_id', 'shop_id']).size().reset_index().rename(
            columns={0: "user_shop_action5_counts_" + suffix})
        user_shop_action1_counts = tmp[(tmp.type == 1)].groupby(['user_id', 'shop_id']).size().reset_index().rename(
            columns={0: "user_shop_action1_counts_" + suffix})
        user_shop_action = pd.merge(user_shop_action1_counts, user_shop_action2_counts, on=['user_id', 'shop_id'],
                                    how='outer')
        user_shop_action = pd.merge(user_shop_action, user_shop_action3_counts, on=['user_id', 'shop_id'], how='outer')
        user_shop_action = pd.merge(user_shop_action, user_shop_action4_counts, on=['user_id', 'shop_id'], how='outer')
        user_shop_action = pd.merge(user_shop_action, user_shop_action5_counts, on=['user_id', 'shop_id'], how='outer')

        sex_sku_action2_counts = tmp[(tmp.type == 2)].groupby(['sex', 'sku_id']).size().reset_index().rename(
            columns={0: "sex_sku_action2_counts_" + suffix})
        sex_sku_action3_counts = tmp[(tmp.type == 3)].groupby(['sex', 'sku_id']).size().reset_index().rename(
            columns={0: "sex_sku_action3_counts_" + suffix})
        sex_sku_action4_counts = tmp[(tmp.type == 4)].groupby(['sex', 'sku_id']).size().reset_index().rename(
            columns={0: "sex_sku_action4_counts_" + suffix})
        sex_sku_action5_counts = tmp[(tmp.type == 5)].groupby(['sex', 'sku_id']).size().reset_index().rename(
            columns={0: "sex_sku_action5_counts_" + suffix})
        sex_sku_action1_counts = tmp[(tmp.type == 1)].groupby(['sex', 'sku_id']).size().reset_index().rename(
            columns={0: "sex_sku_action1_counts_" + suffix})
        sex_sku_action = pd.merge(sex_sku_action1_counts, sex_sku_action2_counts, on=['sex', 'sku_id'], how='outer')
        sex_sku_action = pd.merge(sex_sku_action, sex_sku_action3_counts, on=['sex', 'sku_id'], how='outer')
        sex_sku_action = pd.merge(sex_sku_action, sex_sku_action4_counts, on=['sex', 'sku_id'], how='outer')
        sex_sku_action = pd.merge(sex_sku_action, sex_sku_action5_counts, on=['sex', 'sku_id'], how='outer')
        sex_cate_action2_counts = tmp[(tmp.type == 2)].groupby(['sex', 'cate']).size().reset_index().rename(
            columns={0: "sex_cate_action2_counts_" + suffix})
        sex_cate_action3_counts = tmp[(tmp.type == 3)].groupby(['sex', 'cate']).size().reset_index().rename(
            columns={0: "sex_cate_action3_counts_" + suffix})
        sex_cate_action4_counts = tmp[(tmp.type == 4)].groupby(['sex', 'cate']).size().reset_index().rename(
            columns={0: "sex_cate_action4_counts_" + suffix})
        sex_cate_action5_counts = tmp[(tmp.type == 5)].groupby(['sex', 'cate']).size().reset_index().rename(
            columns={0: "sex_cate_action5_counts_" + suffix})
        sex_cate_action1_counts = tmp[(tmp.type == 1)].groupby(['sex', 'cate']).size().reset_index().rename(
            columns={0: "sex_cate_action1_counts_" + suffix})
        sex_cate_action = pd.merge(sex_cate_action1_counts, sex_cate_action2_counts, on=['sex', 'cate'], how='outer')
        sex_cate_action = pd.merge(sex_cate_action, sex_cate_action3_counts, on=['sex', 'cate'], how='outer')
        sex_cate_action = pd.merge(sex_cate_action, sex_cate_action4_counts, on=['sex', 'cate'], how='outer')
        sex_cate_action = pd.merge(sex_cate_action, sex_cate_action5_counts, on=['sex', 'cate'], how='outer')
        sex_brand_action2_counts = tmp[(tmp.type == 2)].groupby(['sex', 'brand']).size().reset_index().rename(
            columns={0: "sex_brand_action2_counts_" + suffix})
        sex_brand_action3_counts = tmp[(tmp.type == 3)].groupby(['sex', 'brand']).size().reset_index().rename(
            columns={0: "sex_brand_action3_counts_" + suffix})
        sex_brand_action4_counts = tmp[(tmp.type == 4)].groupby(['sex', 'brand']).size().reset_index().rename(
            columns={0: "sex_brand_action4_counts_" + suffix})
        sex_brand_action5_counts = tmp[(tmp.type == 5)].groupby(['sex', 'brand']).size().reset_index().rename(
            columns={0: "sex_brand_action5_counts_" + suffix})
        sex_brand_action1_counts = tmp[(tmp.type == 1)].groupby(['sex', 'brand']).size().reset_index().rename(
            columns={0: "sex_brand_action1_counts_" + suffix})
        sex_brand_action = pd.merge(sex_brand_action1_counts, sex_brand_action2_counts, on=['sex', 'brand'],
                                    how='outer')
        sex_brand_action = pd.merge(sex_brand_action, sex_brand_action3_counts, on=['sex', 'brand'], how='outer')
        sex_brand_action = pd.merge(sex_brand_action, sex_brand_action4_counts, on=['sex', 'brand'], how='outer')
        sex_brand_action = pd.merge(sex_brand_action, sex_brand_action5_counts, on=['sex', 'brand'], how='outer')
        sex_shop_action2_counts = tmp[(tmp.type == 2)].groupby(['sex', 'shop_id']).size().reset_index().rename(
            columns={0: "sex_shop_action2_counts_" + suffix})
        sex_shop_action3_counts = tmp[(tmp.type == 3)].groupby(['sex', 'shop_id']).size().reset_index().rename(
            columns={0: "sex_shop_action3_counts_" + suffix})
        sex_shop_action4_counts = tmp[(tmp.type == 4)].groupby(['sex', 'shop_id']).size().reset_index().rename(
            columns={0: "sex_shop_action4_counts_" + suffix})
        sex_shop_action5_counts = tmp[(tmp.type == 5)].groupby(['sex', 'shop_id']).size().reset_index().rename(
            columns={0: "sex_shop_action5_counts_" + suffix})
        sex_shop_action1_counts = tmp[(tmp.type == 1)].groupby(['sex', 'shop_id']).size().reset_index().rename(
            columns={0: "sex_shop_action1_counts_" + suffix})
        sex_shop_action = pd.merge(sex_shop_action1_counts, sex_shop_action2_counts, on=['sex', 'shop_id'], how='outer')
        sex_shop_action = pd.merge(sex_shop_action, sex_shop_action3_counts, on=['sex', 'shop_id'], how='outer')
        sex_shop_action = pd.merge(sex_shop_action, sex_shop_action4_counts, on=['sex', 'shop_id'], how='outer')
        sex_shop_action = pd.merge(sex_shop_action, sex_shop_action5_counts, on=['sex', 'shop_id'], how='outer')

        age_sku_action2_counts = tmp[(tmp.type == 2)].groupby(['age', 'sku_id']).size().reset_index().rename(
            columns={0: "age_sku_action2_counts_" + suffix})
        age_sku_action3_counts = tmp[(tmp.type == 3)].groupby(['age', 'sku_id']).size().reset_index().rename(
            columns={0: "age_sku_action3_counts_" + suffix})
        age_sku_action4_counts = tmp[(tmp.type == 4)].groupby(['age', 'sku_id']).size().reset_index().rename(
            columns={0: "age_sku_action4_counts_" + suffix})
        age_sku_action5_counts = tmp[(tmp.type == 5)].groupby(['age', 'sku_id']).size().reset_index().rename(
            columns={0: "age_sku_action5_counts_" + suffix})
        age_sku_action1_counts = tmp[(tmp.type == 1)].groupby(['age', 'sku_id']).size().reset_index().rename(
            columns={0: "age_sku_action1_counts_" + suffix})
        age_sku_action = pd.merge(age_sku_action1_counts, age_sku_action2_counts, on=['age', 'sku_id'], how='outer')
        age_sku_action = pd.merge(age_sku_action, age_sku_action3_counts, on=['age', 'sku_id'], how='outer')
        age_sku_action = pd.merge(age_sku_action, age_sku_action4_counts, on=['age', 'sku_id'], how='outer')
        age_sku_action = pd.merge(age_sku_action, age_sku_action5_counts, on=['age', 'sku_id'], how='outer')
        age_cate_action2_counts = tmp[(tmp.type == 2)].groupby(['age', 'cate']).size().reset_index().rename(
            columns={0: "age_cate_action2_counts_" + suffix})
        age_cate_action3_counts = tmp[(tmp.type == 3)].groupby(['age', 'cate']).size().reset_index().rename(
            columns={0: "age_cate_action3_counts_" + suffix})
        age_cate_action4_counts = tmp[(tmp.type == 4)].groupby(['age', 'cate']).size().reset_index().rename(
            columns={0: "age_cate_action4_counts_" + suffix})
        age_cate_action5_counts = tmp[(tmp.type == 5)].groupby(['age', 'cate']).size().reset_index().rename(
            columns={0: "age_cate_action5_counts_" + suffix})
        age_cate_action1_counts = tmp[(tmp.type == 1)].groupby(['age', 'cate']).size().reset_index().rename(
            columns={0: "age_cate_action1_counts_" + suffix})
        age_cate_action = pd.merge(age_cate_action1_counts, age_cate_action2_counts, on=['age', 'cate'], how='outer')
        age_cate_action = pd.merge(age_cate_action, age_cate_action3_counts, on=['age', 'cate'], how='outer')
        age_cate_action = pd.merge(age_cate_action, age_cate_action4_counts, on=['age', 'cate'], how='outer')
        age_cate_action = pd.merge(age_cate_action, age_cate_action5_counts, on=['age', 'cate'], how='outer')
        age_brand_action2_counts = tmp[(tmp.type == 2)].groupby(['age', 'brand']).size().reset_index().rename(
            columns={0: "age_brand_action2_counts_" + suffix})
        age_brand_action3_counts = tmp[(tmp.type == 3)].groupby(['age', 'brand']).size().reset_index().rename(
            columns={0: "age_brand_action3_counts_" + suffix})
        age_brand_action4_counts = tmp[(tmp.type == 4)].groupby(['age', 'brand']).size().reset_index().rename(
            columns={0: "age_brand_action4_counts_" + suffix})
        age_brand_action5_counts = tmp[(tmp.type == 5)].groupby(['age', 'brand']).size().reset_index().rename(
            columns={0: "age_brand_action5_counts_" + suffix})
        age_brand_action1_counts = tmp[(tmp.type == 1)].groupby(['age', 'brand']).size().reset_index().rename(
            columns={0: "age_brand_action1_counts_" + suffix})
        age_brand_action = pd.merge(age_brand_action1_counts, age_brand_action2_counts, on=['age', 'brand'],
                                    how='outer')
        age_brand_action = pd.merge(age_brand_action, age_brand_action3_counts, on=['age', 'brand'], how='outer')
        age_brand_action = pd.merge(age_brand_action, age_brand_action4_counts, on=['age', 'brand'], how='outer')
        age_brand_action = pd.merge(age_brand_action, age_brand_action5_counts, on=['age', 'brand'], how='outer')
        age_shop_action2_counts = tmp[(tmp.type == 2)].groupby(['age', 'shop_id']).size().reset_index().rename(
            columns={0: "age_shop_action2_counts_" + suffix})
        age_shop_action3_counts = tmp[(tmp.type == 3)].groupby(['age', 'shop_id']).size().reset_index().rename(
            columns={0: "age_shop_action3_counts_" + suffix})
        age_shop_action4_counts = tmp[(tmp.type == 4)].groupby(['age', 'shop_id']).size().reset_index().rename(
            columns={0: "age_shop_action4_counts_" + suffix})
        age_shop_action5_counts = tmp[(tmp.type == 5)].groupby(['age', 'shop_id']).size().reset_index().rename(
            columns={0: "age_shop_action5_counts_" + suffix})
        age_shop_action1_counts = tmp[(tmp.type == 1)].groupby(['age', 'shop_id']).size().reset_index().rename(
            columns={0: "age_shop_action1_counts_" + suffix})
        age_shop_action = pd.merge(age_shop_action1_counts, age_shop_action2_counts, on=['age', 'shop_id'], how='outer')
        age_shop_action = pd.merge(age_shop_action, age_shop_action3_counts, on=['age', 'shop_id'], how='outer')
        age_shop_action = pd.merge(age_shop_action, age_shop_action4_counts, on=['age', 'shop_id'], how='outer')
        age_shop_action = pd.merge(age_shop_action, age_shop_action5_counts, on=['age', 'shop_id'], how='outer')

        shop_action = pd.merge(shop_action1_counts, shop_action2_counts, on="shop_id", how="outer")
        shop_action = pd.merge(shop_action, shop_action3_counts, on="shop_id", how="outer")
        shop_action = pd.merge(shop_action, shop_action4_counts, on="shop_id", how="outer")
        shop_action = pd.merge(shop_action, shop_action5_counts, on="shop_id", how="outer")
        user_action = pd.merge(user_action1_counts, user_action2_counts, on="user_id", how="outer")
        user_action = pd.merge(user_action, user_action3_counts, on="user_id", how="outer")
        user_action = pd.merge(user_action, user_action4_counts, on="user_id", how="outer")
        user_action = pd.merge(user_action, user_action5_counts, on="user_id", how="outer")
        user_action = pd.merge(user_action, user_action2_counts_date, on="user_id", how="outer")
        user_action = pd.merge(user_action, user_action3_counts_date, on="user_id", how="outer")
        user_action = pd.merge(user_action, user_action4_counts_date, on="user_id", how="outer")
        cate_action = pd.merge(cate_action1_counts, cate_action2_counts, on="cate", how="outer")
        cate_action = pd.merge(cate_action, cate_action3_counts, on="cate", how="outer")
        cate_action = pd.merge(cate_action, cate_action4_counts, on="cate", how="outer")
        cate_action = pd.merge(cate_action, cate_action5_counts, on="cate", how="outer")
        sku_action = pd.merge(sku_action1_counts, sku_action2_counts, on="sku_id", how="outer")
        sku_action = pd.merge(sku_action, sku_action3_counts, on="sku_id", how="outer")
        sku_action = pd.merge(sku_action, sku_action4_counts, on="sku_id", how="outer")
        sku_action = pd.merge(sku_action, sku_action5_counts, on="sku_id", how="outer")
        province_action = pd.merge(province_action1_counts, province_action2_counts, on="province", how="outer")
        province_action = pd.merge(province_action, province_action3_counts, on="province", how="outer")
        province_action = pd.merge(province_action, province_action4_counts, on="province", how="outer")
        province_action = pd.merge(province_action, province_action5_counts, on="province", how="outer")
        city_action = pd.merge(city_action1_counts, city_action2_counts, on="city", how="outer")
        city_action = pd.merge(city_action, city_action3_counts, on="city", how="outer")
        city_action = pd.merge(city_action, city_action4_counts, on="city", how="outer")
        city_action = pd.merge(city_action, city_action5_counts, on="city", how="outer")
        county_action = pd.merge(county_action1_counts, county_action2_counts, on="county", how="outer")
        county_action = pd.merge(county_action, county_action3_counts, on="county", how="outer")
        county_action = pd.merge(county_action, county_action4_counts, on="county", how="outer")
        county_action = pd.merge(county_action, county_action5_counts, on="county", how="outer")
        brand_action = pd.merge(brand_action1_counts, brand_action2_counts, on="brand", how="outer")
        brand_action = pd.merge(brand_action, brand_action3_counts, on="brand", how="outer")
        brand_action = pd.merge(brand_action, brand_action4_counts, on="brand", how="outer")
        brand_action = pd.merge(brand_action, brand_action5_counts, on="brand", how="outer")
        age_action = pd.merge(age_action1_counts, age_action2_counts, on="age", how="outer")
        age_action = pd.merge(age_action, age_action3_counts, on="age", how="outer")
        age_action = pd.merge(age_action, age_action4_counts, on="age", how="outer")
        age_action = pd.merge(age_action, age_action5_counts, on="age", how="outer")
        sex_action = pd.merge(sex_action1_counts, sex_action2_counts, on="sex", how="outer")
        sex_action = pd.merge(sex_action, sex_action3_counts, on="sex", how="outer")
        sex_action = pd.merge(sex_action, sex_action4_counts, on="sex", how="outer")
        sex_action = pd.merge(sex_action, sex_action5_counts, on="sex", how="outer")

        sku_action['action_date'] = dat
        shop_action['action_date'] = dat
        user_action['action_date'] = dat
        cate_action['action_date'] = dat
        province_action['action_date'] = dat
        city_action['action_date'] = dat
        county_action['action_date'] = dat
        brand_action['action_date'] = dat
        age_action['action_date'] = dat
        sex_action['action_date'] = dat

        user_sku_action['action_date'] = dat
        user_cate_action['action_date'] = dat
        user_brand_action['action_date'] = dat
        user_shop_action['action_date'] = dat
        sex_sku_action['action_date'] = dat
        sex_cate_action['action_date'] = dat
        sex_brand_action['action_date'] = dat
        sex_shop_action['action_date'] = dat
        age_sku_action['action_date'] = dat
        age_cate_action['action_date'] = dat
        age_brand_action['action_date'] = dat
        age_shop_action['action_date'] = dat

        return shop_action, user_action, cate_action, sku_action, province_action, city_action, county_action, brand_action, age_action, sex_action \
            , user_sku_action, user_cate_action, user_brand_action, user_shop_action, sex_sku_action, sex_cate_action, sex_brand_action, sex_shop_action, age_sku_action, age_cate_action, age_brand_action, age_shop_action

    def add_user_behavier_fea_day(self,sample,days):
        days_dict={
            '1day':["2018-04-08","2018-04-09","2018-04-10","2018-04-11","2018-04-12","2018-04-13","2018-04-14","2018-04-15"],
            '3day':["2018-04-06","2018-04-07","2018-04-08","2018-04-09","2018-04-10","2018-04-11","2018-04-12","2018-04-13"],
            '7day':["2018-04-02","2018-04-03","2018-04-04","2018-04-05","2018-04-06","2018-04-07","2018-04-08","2018-04-09"],
            '21day':["2018-03-19","2018-03-20","2018-03-21","2018-03-22","2018-03-23","2018-03-24","2018-03-25","2018-03-26"]
        }
        shop_action0408, user_action0408, cate_action0408, sku_action0408, province_action0408, city_action0408, county_action0408, brand_action0408, age_action0408, sex_action0408 \
            , user_sku_action0408, user_cate_action0408, user_brand_action0408, user_shop_action0408, sex_sku_action0408, sex_cate_action0408, sex_brand_action0408, sex_shop_action0408 \
            , age_sku_action0408, age_cate_action0408, age_brand_action0408, age_shop_action0408 = self.get_user_behavier_fea(
            "2018-04-08", days_dict[days][0], days)
        shop_action0409, user_action0409, cate_action0409, sku_action0409, province_action0409, city_action0409, county_action0409, brand_action0409, age_action0409, sex_action0409 \
            , user_sku_action0409, user_cate_action0409, user_brand_action0409, user_shop_action0409, sex_sku_action0409, sex_cate_action0409, sex_brand_action0409, sex_shop_action0409 \
            , age_sku_action0409, age_cate_action0409, age_brand_action0409, age_shop_action0409 = self.get_user_behavier_fea(
            "2018-04-09", days_dict[days][1], days)
        shop_action0410, user_action0410, cate_action0410, sku_action0410, province_action0410, city_action0410, county_action0410, brand_action0410, age_action0410, sex_action0410 \
            , user_sku_action0410, user_cate_action0410, user_brand_action0410, user_shop_action0410, sex_sku_action0410, sex_cate_action0410, sex_brand_action0410, sex_shop_action0410 \
            , age_sku_action0410, age_cate_action0410, age_brand_action0410, age_shop_action0410 = self.get_user_behavier_fea(
            "2018-04-10", days_dict[days][2], days)
        shop_action0411, user_action0411, cate_action0411, sku_action0411, province_action0411, city_action0411, county_action0411, brand_action0411, age_action0411, sex_action0411 \
            , user_sku_action0411, user_cate_action0411, user_brand_action0411, user_shop_action0411, sex_sku_action0411, sex_cate_action0411, sex_brand_action0411, sex_shop_action0411 \
            , age_sku_action0411, age_cate_action0411, age_brand_action0411, age_shop_action0411 = self.get_user_behavier_fea(
            "2018-04-11", days_dict[days][3], days)
        shop_action0412, user_action0412, cate_action0412, sku_action0412, province_action0412, city_action0412, county_action0412, brand_action0412, age_action0412, sex_action0412 \
            , user_sku_action0412, user_cate_action0412, user_brand_action0412, user_shop_action0412, sex_sku_action0412, sex_cate_action0412, sex_brand_action0412, sex_shop_action0412 \
            , age_sku_action0412, age_cate_action0412, age_brand_action0412, age_shop_action0412 = self.get_user_behavier_fea(
            "2018-04-12", days_dict[days][4], days)
        shop_action0413, user_action0413, cate_action0413, sku_action0413, province_action0413, city_action0413, county_action0413, brand_action0413, age_action0413, sex_action0413 \
            , user_sku_action0413, user_cate_action0413, user_brand_action0413, user_shop_action0413, sex_sku_action0413, sex_cate_action0413, sex_brand_action0413, sex_shop_action0413 \
            , age_sku_action0413, age_cate_action0413, age_brand_action0413, age_shop_action0413 = self.get_user_behavier_fea(
            "2018-04-13", days_dict[days][5], days)
        shop_action0414, user_action0414, cate_action0414, sku_action0414, province_action0414, city_action0414, county_action0414, brand_action0414, age_action0414, sex_action0414 \
            , user_sku_action0414, user_cate_action0414, user_brand_action0414, user_shop_action0414, sex_sku_action0414, sex_cate_action0414, sex_brand_action0414, sex_shop_action0414 \
            , age_sku_action0414, age_cate_action0414, age_brand_action0414, age_shop_action0414 = self.get_user_behavier_fea(
            "2018-04-14", days_dict[days][6], days)
        shop_action0415, user_action0415, cate_action0415, sku_action0415, province_action0415, city_action0415, county_action0415, brand_action0415, age_action0415, sex_action0415 \
            , user_sku_action0415, user_cate_action0415, user_brand_action0415, user_shop_action0415, sex_sku_action0415, sex_cate_action0415, sex_brand_action0415, sex_shop_action0415 \
            , age_sku_action0415, age_cate_action0415, age_brand_action0415, age_shop_action0415 = self.get_user_behavier_fea(
            "2018-04-15", days_dict[days][7], days)

        shop_action = pd.concat([shop_action0408, shop_action0409, shop_action0410, shop_action0411, shop_action0412, shop_action0413 , shop_action0414, shop_action0415])
        user_action = pd.concat([user_action0408, user_action0409, user_action0410, user_action0411, user_action0412, user_action0413, user_action0414, user_action0415])
        cate_action = pd.concat([cate_action0408, cate_action0409, cate_action0410, cate_action0411, cate_action0412, cate_action0413, cate_action0414, cate_action0415])
        # sku_action = pd.concat([sku_action0408, sku_action0409, sku_action0410, sku_action0411, sku_action0412, sku_action0413, sku_action0414, sku_action0415])
        province_action = pd.concat([province_action0408, province_action0409, province_action0410, province_action0411, province_action0412, province_action0413, province_action0414, province_action0415])
        city_action = pd.concat([city_action0408, city_action0409, city_action0410, city_action0411, city_action0412, city_action0413, city_action0414, city_action0415])
        county_action = pd.concat([county_action0408, county_action0409, county_action0410, county_action0411, county_action0412, county_action0413, county_action0414, county_action0415])
        # brand_action = pd.concat([brand_action0408, brand_action0409, brand_action0410, brand_action0411, brand_action0412, brand_action0413, brand_action0414, brand_action0415])
        age_action = pd.concat([age_action0408, age_action0409, age_action0410, age_action0411, age_action0412, age_action0413, age_action0414, age_action0415])
        sex_action = pd.concat([sex_action0408, sex_action0409, sex_action0410, sex_action0411, sex_action0412, sex_action0413, sex_action0414, sex_action0415])
        # user_sku_action = pd.concat([user_sku_action0408, user_sku_action0409, user_sku_action0410, user_sku_action0411, user_sku_action0412, user_sku_action0413, user_sku_action0414, user_sku_action0415])
        user_cate_action = pd.concat([user_cate_action0408, user_cate_action0409, user_cate_action0410, user_cate_action0411,user_cate_action0412, user_cate_action0413, user_cate_action0414,user_cate_action0415])
        # user_brand_action = pd.concat([user_brand_action0408, user_brand_action0409, user_brand_action0410, user_brand_action0411,user_brand_action0412, user_brand_action0413, user_brand_action0414,user_brand_action0415])
        user_shop_action = pd.concat([user_shop_action0408, user_shop_action0409, user_shop_action0410, user_shop_action0411,user_shop_action0412, user_shop_action0413, user_shop_action0414,user_shop_action0415])
        # sex_sku_action = pd.concat([sex_sku_action0408, sex_sku_action0409, sex_sku_action0410, sex_sku_action0411, sex_sku_action0412, sex_sku_action0413, sex_sku_action0414, sex_sku_action0415])
        sex_cate_action = pd.concat([sex_cate_action0408, sex_cate_action0409, sex_cate_action0410, sex_cate_action0411, sex_cate_action0412, sex_cate_action0413, sex_cate_action0414, sex_cate_action0415])
        # sex_brand_action = pd.concat([sex_brand_action0408, sex_brand_action0409, sex_brand_action0410, sex_brand_action0411,sex_brand_action0412, sex_brand_action0413, sex_brand_action0414,sex_brand_action0415])
        sex_shop_action = pd.concat([sex_shop_action0408, sex_shop_action0409, sex_shop_action0410, sex_shop_action0411, sex_shop_action0412, sex_shop_action0413, sex_shop_action0414, sex_shop_action0415])
        # age_sku_action = pd.concat([age_sku_action0408, age_sku_action0409, age_sku_action0410, age_sku_action0411, age_sku_action0412, age_sku_action0413, age_sku_action0414, age_sku_action0415])
        age_cate_action = pd.concat([age_cate_action0408, age_cate_action0409, age_cate_action0410, age_cate_action0411, age_cate_action0412, age_cate_action0413, age_cate_action0414, age_cate_action0415])
        # age_brand_action = pd.concat([age_brand_action0408, age_brand_action0409, age_brand_action0410, age_brand_action0411, age_brand_action0412, age_brand_action0413, age_brand_action0414, age_brand_action0415])
        age_shop_action = pd.concat( [age_shop_action0408, age_shop_action0409, age_shop_action0410, age_shop_action0411, age_shop_action0412, age_shop_action0413, age_shop_action0414, age_shop_action0415])

        sample = pd.merge(sample, shop_action, on=['shop_id', 'action_date'], how='left')
        sample = pd.merge(sample, user_action, on=['user_id', 'action_date'])
        sample = pd.merge(sample, cate_action, on=['cate', 'action_date'])
        sample = pd.merge(sample, province_action, on=['province', 'action_date'])
        sample = pd.merge(sample, city_action, on=['city', 'action_date'])
        sample = pd.merge(sample, county_action, on=['county', 'action_date'])
        sample = pd.merge(sample, age_action, on=['age', 'action_date'])
        sample = pd.merge(sample, sex_action, on=['sex', 'action_date'])
        sample = pd.merge(sample, user_cate_action, on=['user_id', 'cate', 'action_date'], how='left')
        sample = pd.merge(sample, user_shop_action, on=['user_id', 'shop_id', 'action_date'], how='left')
        sample = pd.merge(sample, sex_cate_action, on=['sex', 'cate', 'action_date'], how='left')
        sample = pd.merge(sample, sex_shop_action, on=['sex', 'shop_id', 'action_date'], how='left')
        sample = pd.merge(sample, age_cate_action, on=['age', 'cate', 'action_date'], how='left')
        sample = pd.merge(sample, age_shop_action, on=['age', 'shop_id', 'action_date'], how='left')
        return sample

    def add_user_behavier_fea(self, sample):
        for days in ['1day','3day','7day','21day']:
            sample = self.add_user_behavier_fea_day(sample,days)
            print('user behavier 1'+days+' finished')
        return sample

    def add_static_action_fea(self, sample):
        tmp = self.jdata_action[self.jdata_action.action_date < '2018-03-26']
        user_action_times = tmp.groupby(['user_id']).size().reset_index().rename(columns={0: 'user_action_times'})
        user_action4_times = tmp[tmp.type == 4].groupby(['user_id']).size().reset_index().rename(
            columns={0: 'user_action4_times'})
        user_action3_times = tmp[tmp.type == 3].groupby(['user_id']).size().reset_index().rename(
            columns={0: 'user_action3_times'})
        user_action2_times = tmp[tmp.type == 2].groupby(['user_id']).size().reset_index().rename(
            columns={0: 'user_action2_times'})
        user_action1_times = tmp[tmp.type == 1].groupby(['user_id']).size().reset_index().rename(
            columns={0: 'user_action1_times'})
        user_action_times = pd.merge(user_action_times, user_action1_times, on='user_id', how='outer')
        user_action_times = pd.merge(user_action_times, user_action2_times, on='user_id', how='outer')
        user_action_times = pd.merge(user_action_times, user_action3_times, on='user_id', how='outer')
        user_action_times = pd.merge(user_action_times, user_action4_times, on='user_id', how='outer')
        user_action_times = user_action_times.fillna(0)
        sample = pd.merge(sample, user_action_times, on='user_id', how='left')

        shop_action_times = tmp.groupby(['shop_id']).size().reset_index().rename(columns={0: 'shop_action_times'})
        shop_action4_times = tmp[tmp.type == 4].groupby(['shop_id']).size().reset_index().rename(
            columns={0: 'shop_action4_times'})
        shop_action3_times = tmp[tmp.type == 3].groupby(['shop_id']).size().reset_index().rename(
            columns={0: 'shop_action3_times'})
        shop_action2_times = tmp[tmp.type == 2].groupby(['shop_id']).size().reset_index().rename(
            columns={0: 'shop_action2_times'})
        shop_action1_times = tmp[tmp.type == 1].groupby(['shop_id']).size().reset_index().rename(
            columns={0: 'shop_action1_times'})
        shop_action_times = pd.merge(shop_action_times, shop_action1_times, on='shop_id', how='outer')
        shop_action_times = pd.merge(shop_action_times, shop_action2_times, on='shop_id', how='outer')
        shop_action_times = pd.merge(shop_action_times, shop_action3_times, on='shop_id', how='outer')
        shop_action_times = pd.merge(shop_action_times, shop_action4_times, on='shop_id', how='outer')
        shop_action_times = shop_action_times.fillna(0)
        sample = pd.merge(sample, shop_action_times, on='shop_id', how='left')

        cate_action_times = tmp.groupby(['cate']).size().reset_index().rename(columns={0: 'cate_action_times'})
        cate_action4_times = tmp[tmp.type == 4].groupby(['cate']).size().reset_index().rename(
            columns={0: 'cate_action4_times'})
        cate_action3_times = tmp[tmp.type == 3].groupby(['cate']).size().reset_index().rename(
            columns={0: 'cate_action3_times'})
        cate_action2_times = tmp[tmp.type == 2].groupby(['cate']).size().reset_index().rename(
            columns={0: 'cate_action2_times'})
        cate_action1_times = tmp[tmp.type == 1].groupby(['cate']).size().reset_index().rename(
            columns={0: 'cate_action1_times'})
        cate_action_times = pd.merge(cate_action_times, cate_action1_times, on='cate', how='outer')
        cate_action_times = pd.merge(cate_action_times, cate_action2_times, on='cate', how='outer')
        cate_action_times = pd.merge(cate_action_times, cate_action3_times, on='cate', how='outer')
        cate_action_times = pd.merge(cate_action_times, cate_action4_times, on='cate', how='outer')
        cate_action_times = cate_action_times.fillna(0)
        sample = pd.merge(sample, cate_action_times, on='cate', how='left')

        user_cate_action_times = tmp.groupby(['user_id', 'cate']).size().reset_index().rename(
            columns={0: 'user_cate_action_times'})
        user_cate_action4_times = tmp[tmp.type == 4].groupby(['user_id', 'cate']).size().reset_index().rename(
            columns={0: 'user_cate_action4_times'})
        user_cate_action3_times = tmp[tmp.type == 3].groupby(['user_id', 'cate']).size().reset_index().rename(
            columns={0: 'user_cate_action3_times'})
        user_cate_action2_times = tmp[tmp.type == 2].groupby(['user_id', 'cate']).size().reset_index().rename(
            columns={0: 'user_cate_action2_times'})
        user_cate_action1_times = tmp[tmp.type == 1].groupby(['user_id', 'cate']).size().reset_index().rename(
            columns={0: 'user_cate_action1_times'})
        user_cate_action_times = pd.merge(user_cate_action_times, user_cate_action1_times, on=['user_id', 'cate'],how='outer')
        user_cate_action_times = pd.merge(user_cate_action_times, user_cate_action2_times, on=['user_id', 'cate'],how='outer')
        user_cate_action_times = pd.merge(user_cate_action_times, user_cate_action3_times, on=['user_id', 'cate'],how='outer')
        user_cate_action_times = pd.merge(user_cate_action_times, user_cate_action4_times, on=['user_id', 'cate'],how='outer')
        user_cate_action_times = user_cate_action_times.fillna(0)
        sample = pd.merge(sample, user_cate_action_times, on=['user_id', 'cate'], how='left')

        shop_cate_action_times = tmp.groupby(['shop_id', 'cate']).size().reset_index().rename(columns={0: 'shop_cate_action_times'})
        # shop_cate_action5_times = tmp[tmp.type == 5].groupby(['shop_id', 'cate']).size().reset_index().rename(
        #     columns={0: 'shop_cate_action5_times'})
        shop_cate_action4_times = tmp[tmp.type == 4].groupby(['shop_id', 'cate']).size().reset_index().rename(
            columns={0: 'shop_cate_action4_times'})
        shop_cate_action3_times = tmp[tmp.type == 3].groupby(['shop_id', 'cate']).size().reset_index().rename(
            columns={0: 'shop_cate_action3_times'})
        shop_cate_action2_times = tmp[tmp.type == 2].groupby(['shop_id', 'cate']).size().reset_index().rename(
            columns={0: 'shop_cate_action2_times'})
        shop_cate_action1_times = tmp[tmp.type == 1].groupby(['shop_id', 'cate']).size().reset_index().rename(
            columns={0: 'shop_cate_action1_times'})
        shop_cate_action_times = pd.merge(shop_cate_action_times, shop_cate_action1_times, on=['shop_id', 'cate'],how='outer')
        shop_cate_action_times = pd.merge(shop_cate_action_times, shop_cate_action2_times, on=['shop_id', 'cate'],how='outer')
        shop_cate_action_times = pd.merge(shop_cate_action_times, shop_cate_action3_times, on=['shop_id', 'cate'],how='outer')
        shop_cate_action_times = pd.merge(shop_cate_action_times, shop_cate_action4_times, on=['shop_id', 'cate'],how='outer')
        # shop_cate_action_times = pd.merge(shop_cate_action_times, shop_cate_action5_times, on=['shop_id', 'cate'],how='outer')
        shop_cate_action_times = shop_cate_action_times.fillna(0)
        sample = pd.merge(sample, shop_cate_action_times, on=['shop_id', 'cate'], how='left')

        user_shop_action_times = tmp.groupby(['user_id', 'shop_id']).size().reset_index().rename(columns={0: 'user_shop_action_times'})
        # user_shop_action5_times = tmp[tmp.type == 5].groupby(['user_id', 'shop_id']).size().reset_index().rename(columns={0: 'user_shop_action5_times'})
        user_shop_action4_times = tmp[tmp.type == 4].groupby(['user_id', 'shop_id']).size().reset_index().rename(columns={0: 'user_shop_action4_times'})
        user_shop_action3_times = tmp[tmp.type == 3].groupby(['user_id', 'shop_id']).size().reset_index().rename(columns={0: 'user_shop_action3_times'})
        user_shop_action2_times = tmp[tmp.type == 2].groupby(['user_id', 'shop_id']).size().reset_index().rename(columns={0: 'user_shop_action2_times'})
        user_shop_action1_times = tmp[tmp.type == 1].groupby(['user_id', 'shop_id']).size().reset_index().rename(columns={0: 'user_shop_action1_times'})
        user_shop_action_times = pd.merge(user_shop_action_times, user_shop_action1_times, on=['user_id', 'shop_id'], how='outer')
        user_shop_action_times = pd.merge(user_shop_action_times, user_shop_action2_times, on=['user_id', 'shop_id'], how='outer')
        user_shop_action_times = pd.merge(user_shop_action_times, user_shop_action3_times, on=['user_id', 'shop_id'], how='outer')
        user_shop_action_times = pd.merge(user_shop_action_times, user_shop_action4_times, on=['user_id', 'shop_id'], how='outer')
        # user_shop_action_times = pd.merge(user_shop_action_times, user_shop_action5_times, on=['user_id', 'shop_id'],how='outer')
        user_shop_action_times = user_shop_action_times.fillna(0)
        sample = pd.merge(sample, user_shop_action_times, on=['user_id', 'shop_id'], how='left')

        user_cate_shop_action_times = tmp.groupby(['user_id', 'cate', 'shop_id']).size().reset_index().rename( columns={0: 'user_cate_shop_action_times'})
        # user_cate_shop_action5_times = tmp[tmp.type == 5].groupby(['user_id', 'cate', 'shop_id']).size().reset_index().rename(columns={0: 'user_cate_shop_action5_times'})
        user_cate_shop_action4_times = tmp[tmp.type == 4].groupby(['user_id', 'cate', 'shop_id']).size().reset_index().rename(columns={0: 'user_cate_shop_action4_times'})
        user_cate_shop_action3_times = tmp[tmp.type == 3].groupby(['user_id', 'cate', 'shop_id']).size().reset_index().rename(columns={0: 'user_cate_shop_action3_times'})
        user_cate_shop_action2_times = tmp[tmp.type == 2].groupby(['user_id', 'cate', 'shop_id']).size().reset_index().rename(columns={0: 'user_cate_shop_action2_times'})
        user_cate_shop_action1_times = tmp[tmp.type == 1].groupby(['user_id', 'cate', 'shop_id']).size().reset_index().rename(columns={0: 'user_cate_shop_action1_times'})
        user_cate_shop_action_times = pd.merge(user_cate_shop_action_times, user_cate_shop_action1_times,on=['user_id', 'cate', 'shop_id'], how='outer')
        user_cate_shop_action_times = pd.merge(user_cate_shop_action_times, user_cate_shop_action2_times,on=['user_id', 'cate', 'shop_id'], how='outer')
        user_cate_shop_action_times = pd.merge(user_cate_shop_action_times, user_cate_shop_action3_times,on=['user_id', 'cate', 'shop_id'], how='outer')
        user_cate_shop_action_times = pd.merge(user_cate_shop_action_times, user_cate_shop_action4_times,on=['user_id', 'cate', 'shop_id'], how='outer')
        # user_cate_shop_action_times = pd.merge(user_cate_shop_action_times, user_cate_shop_action5_times,on=['user_id', 'cate', 'shop_id'], how='outer')
        user_cate_shop_action_times = user_cate_shop_action_times.fillna(0)
        sample = pd.merge(sample, user_cate_shop_action_times, on=['user_id', 'cate', 'shop_id'], how='left')

        return sample

    # 用户的动态行为特征
    def get_user_behavier_fea2(self, dat, before_dat, suffix):
        tmp = self.jdata_action[(self.jdata_action.action_date >= before_dat) & (self.jdata_action.action_date <= dat)]
        # 用户粒度过去操作的cate种类数
        user_action1_cates = tmp[(tmp.type == 1)].groupby('user_id')['cate'].nunique().reset_index().rename(
            columns={'cate': "user_action1_cates_" + suffix})
        user_action2_cates = tmp[(tmp.type == 2)].groupby('user_id')['cate'].nunique().reset_index().rename(
            columns={'cate': "user_action2_cates_" + suffix})
        user_action3_cates = tmp[(tmp.type == 3)].groupby('user_id')['cate'].nunique().reset_index().rename(
            columns={'cate': "user_action3_cates_" + suffix})
        user_action4_cates = tmp[(tmp.type == 4)].groupby('user_id')['cate'].nunique().reset_index().rename(
            columns={'cate': "user_action4_cates_" + suffix})
        user_action_cates = tmp.groupby('user_id')['cate'].nunique().reset_index().rename(
            columns={'cate': "user_action_cates_" + suffix})
        user_action_cates = pd.merge(user_action_cates, user_action1_cates, on="user_id", how="outer")
        user_action_cates = pd.merge(user_action_cates, user_action2_cates, on="user_id", how="outer")
        user_action_cates = pd.merge(user_action_cates, user_action3_cates, on="user_id", how="outer")
        user_action_cates = pd.merge(user_action_cates, user_action4_cates, on="user_id", how="outer")
        del user_action1_cates, user_action2_cates, user_action3_cates, user_action4_cates
        gc.collect()
        # 用户粒度过去逛过的商店数
        user_action1_shops = tmp[tmp.type == 1].groupby('user_id')['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "user_action1_shops_" + suffix})
        user_action2_shops = tmp[tmp.type == 2].groupby('user_id')['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "user_action2_shops_" + suffix})
        user_action3_shops = tmp[tmp.type == 3].groupby('user_id')['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "user_action3_shops_" + suffix})
        user_action4_shops = tmp[tmp.type == 4].groupby('user_id')['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "user_action4_shops_" + suffix})
        user_action_shops = tmp.groupby('user_id')['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "user_action_shops_" + suffix})
        user_action_shops = pd.merge(user_action_shops, user_action1_shops, on="user_id", how="outer")
        user_action_shops = pd.merge(user_action_shops, user_action2_shops, on="user_id", how="outer")
        user_action_shops = pd.merge(user_action_shops, user_action3_shops, on="user_id", how="outer")
        user_action_shops = pd.merge(user_action_shops, user_action4_shops, on="user_id", how="outer")
        del user_action1_shops, user_action2_shops, user_action3_shops, user_action4_shops
        gc.collect()
        # 用户粒度过去参与过的商品数
        user_action1_skus = tmp[tmp.type == 1].groupby('user_id')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "user_action1_skus_" + suffix})
        user_action2_skus = tmp[tmp.type == 2].groupby('user_id')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "user_action2_skus_" + suffix})
        user_action3_skus = tmp[tmp.type == 3].groupby('user_id')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "user_action3_skus_" + suffix})
        user_action4_skus = tmp[tmp.type == 4].groupby('user_id')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "user_action4_skus_" + suffix})
        user_action_skus = tmp.groupby('user_id')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "user_action_skus_" + suffix})
        user_action_skus = pd.merge(user_action_skus, user_action1_skus, on="user_id", how="outer")
        user_action_skus = pd.merge(user_action_skus, user_action2_skus, on="user_id", how="outer")
        user_action_skus = pd.merge(user_action_skus, user_action3_skus, on="user_id", how="outer")
        user_action_skus = pd.merge(user_action_skus, user_action4_skus, on="user_id", how="outer")
        del user_action1_skus, user_action2_skus, user_action3_skus, user_action4_skus
        gc.collect()

        # 用户粒度过去module_ids数量
        user_action1_modules = tmp[tmp.type == 1].groupby('user_id')['module_id'].nunique().reset_index().rename(
            columns={'module_id': "user_action1_modules_" + suffix})
        user_action2_modules = tmp[tmp.type == 2].groupby('user_id')['module_id'].nunique().reset_index().rename(
            columns={'module_id': "user_action2_modules_" + suffix})
        user_action3_modules = tmp[tmp.type == 3].groupby('user_id')['module_id'].nunique().reset_index().rename(
            columns={'module_id': "user_action3_modules_" + suffix})
        user_action4_modules = tmp[tmp.type == 4].groupby('user_id')['module_id'].nunique().reset_index().rename(
            columns={'module_id': "user_action4_modules_" + suffix})
        user_action_modules = tmp.groupby('user_id')['module_id'].nunique().reset_index().rename(
            columns={'module_id': "user_action_modules_" + suffix})
        user_action_modules = pd.merge(user_action_modules, user_action1_modules, on="user_id", how="outer")
        user_action_modules = pd.merge(user_action_modules, user_action2_modules, on="user_id", how="outer")
        user_action_modules = pd.merge(user_action_modules, user_action3_modules, on="user_id", how="outer")
        user_action_modules = pd.merge(user_action_modules, user_action4_modules, on="user_id", how="outer")
        del user_action1_modules, user_action2_modules, user_action3_modules, user_action4_modules
        gc.collect()

        user_action_times = pd.merge(user_action_cates, user_action_modules, on='user_id', how='left')
        user_action_times = pd.merge(user_action_times, user_action_shops, on='user_id', how='left')
        user_action_times = pd.merge(user_action_times, user_action_skus, on='user_id', how='left')
        del user_action_cates, user_action_modules, user_action_shops, user_action_skus
        gc.collect()

        # 品类 该品类下 被浏览的商品个数 品牌个数 商店
        cate_action1_shops = tmp[tmp.type == 1].groupby('cate')['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "cate_action1_shops_" + suffix})
        cate_action2_shops = tmp[tmp.type == 2].groupby('cate')['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "cate_action2_shops_" + suffix})
        cate_action3_shops = tmp[tmp.type == 3].groupby('cate')['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "cate_action3_shops_" + suffix})
        cate_action4_shops = tmp[tmp.type == 4].groupby('cate')['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "cate_action4_shops_" + suffix})
        cate_action_shops = tmp.groupby('cate')['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "cate_action_shops_" + suffix})
        cate_action1_skus = tmp[tmp.type == 1].groupby('cate')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "cate_action1_skus_" + suffix})
        cate_action2_skus = tmp[tmp.type == 2].groupby('cate')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "cate_action2_skus_" + suffix})
        cate_action3_skus = tmp[tmp.type == 3].groupby('cate')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "cate_action3_skus_" + suffix})
        cate_action4_skus = tmp[tmp.type == 4].groupby('cate')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "cate_action4_skus_" + suffix})
        cate_action_skus = tmp.groupby('cate')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "cate_action_skus_" + suffix})
        cate_action1_brands = tmp[tmp.type == 1].groupby('cate')['brand'].nunique().reset_index().rename(
            columns={'brand': "cate_action1_brands_" + suffix})
        cate_action2_brands = tmp[tmp.type == 2].groupby('cate')['brand'].nunique().reset_index().rename(
            columns={'brand': "cate_action2_brands_" + suffix})
        cate_action3_brands = tmp[tmp.type == 3].groupby('cate')['brand'].nunique().reset_index().rename(
            columns={'brand': "cate_action3_brands_" + suffix})
        cate_action4_brands = tmp[tmp.type == 4].groupby('cate')['brand'].nunique().reset_index().rename(
            columns={'brand': "cate_action4_brands_" + suffix})
        cate_action_brands = tmp.groupby('cate')['brand'].nunique().reset_index().rename(
            columns={'brand': "cate_action_brands_" + suffix})
        cate_action_times = reduce(lambda x, y: pd.merge(x, y, on='cate', how='outer'),
                                   [cate_action1_shops, cate_action2_shops, cate_action3_shops, cate_action4_shops,
                                    cate_action_shops, \
                                    cate_action1_skus, cate_action2_skus, cate_action3_skus, cate_action4_skus,
                                    cate_action_skus, \
                                    cate_action1_brands, cate_action2_brands, cate_action3_brands, cate_action4_brands,
                                    cate_action_brands])
        del cate_action1_shops, cate_action2_shops, cate_action3_shops, cate_action4_shops, cate_action_shops, \
            cate_action1_skus, cate_action2_skus, cate_action3_skus, cate_action4_skus, cate_action_skus, \
            cate_action1_brands, cate_action2_brands, cate_action3_brands, cate_action4_brands, cate_action_brands
        gc.collect()
        # 该商店 被浏览的 商品数 品牌数等。
        shop_action1_skus = tmp[tmp.type == 1].groupby('shop_id')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "shop_action1_skus_" + suffix})
        shop_action2_skus = tmp[tmp.type == 2].groupby('shop_id')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "shop_action2_skus_" + suffix})
        shop_action3_skus = tmp[tmp.type == 3].groupby('shop_id')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "shop_action3_skus_" + suffix})
        shop_action4_skus = tmp[tmp.type == 4].groupby('shop_id')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "shop_action4_skus_" + suffix})
        shop_action_skus = tmp.groupby('shop_id')['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "shop_action_skus_" + suffix})
        shop_action1_brands = tmp[tmp.type == 1].groupby('shop_id')['brand'].nunique().reset_index().rename(
            columns={'brand': "shop_action1_brands_" + suffix})
        shop_action2_brands = tmp[tmp.type == 2].groupby('shop_id')['brand'].nunique().reset_index().rename(
            columns={'brand': "shop_action2_brands_" + suffix})
        shop_action3_brands = tmp[tmp.type == 3].groupby('shop_id')['brand'].nunique().reset_index().rename(
            columns={'brand': "shop_action3_brands_" + suffix})
        shop_action4_brands = tmp[tmp.type == 4].groupby('shop_id')['brand'].nunique().reset_index().rename(
            columns={'brand': "shop_action4_brands_" + suffix})
        shop_action_brands = tmp.groupby('shop_id')['brand'].nunique().reset_index().rename(
            columns={'brand': "shop_action_brands_" + suffix})
        shop_action_times = reduce(lambda x, y: pd.merge(x, y, on='shop_id', how='outer'), [shop_action1_skus, shop_action2_skus, shop_action3_skus, shop_action4_skus,
                                    shop_action_skus,
                                    shop_action1_brands, shop_action2_brands, shop_action3_brands, shop_action4_brands,
                                    shop_action_brands])
        del shop_action1_skus, shop_action2_skus, shop_action3_skus, shop_action4_skus, shop_action_skus,shop_action1_brands, shop_action2_brands, shop_action3_brands, shop_action4_brands, shop_action_brands
        gc.collect()
        # 该用户和品类 下浏览的 商店 商品
        user_cate_action1_shops = tmp[tmp.type == 1].groupby(['user_id', 'cate'])[
            'shop_id'].nunique().reset_index().rename(columns={'shop_id': "user_cate_action1_shops_" + suffix})
        user_cate_action2_shops = tmp[tmp.type == 2].groupby(['user_id', 'cate'])[
            'shop_id'].nunique().reset_index().rename(columns={'shop_id': "user_cate_action2_shops_" + suffix})
        user_cate_action3_shops = tmp[tmp.type == 3].groupby(['user_id', 'cate'])[
            'shop_id'].nunique().reset_index().rename(columns={'shop_id': "user_cate_action3_shops_" + suffix})
        user_cate_action4_shops = tmp[tmp.type == 4].groupby(['user_id', 'cate'])[
            'shop_id'].nunique().reset_index().rename(columns={'shop_id': "user_cate_action4_shops_" + suffix})
        user_cate_action_shops = tmp.groupby(['user_id', 'cate'])['shop_id'].nunique().reset_index().rename(
            columns={'shop_id': "user_cate_action_shops_" + suffix})
        user_cate_action1_skus = tmp[tmp.type == 1].groupby(['user_id', 'cate'])[
            'sku_id'].nunique().reset_index().rename(columns={'sku_id': "user_cate_action1_skus_" + suffix})
        user_cate_action2_skus = tmp[tmp.type == 2].groupby(['user_id', 'cate'])[
            'sku_id'].nunique().reset_index().rename(columns={'sku_id': "user_cate_action2_skus_" + suffix})
        user_cate_action3_skus = tmp[tmp.type == 3].groupby(['user_id', 'cate'])[
            'sku_id'].nunique().reset_index().rename(columns={'sku_id': "user_cate_action3_skus_" + suffix})
        user_cate_action4_skus = tmp[tmp.type == 4].groupby(['user_id', 'cate'])[
            'sku_id'].nunique().reset_index().rename(columns={'sku_id': "user_cate_action4_skus_" + suffix})
        user_cate_action_skus = tmp.groupby(['user_id', 'cate'])['sku_id'].nunique().reset_index().rename(
            columns={'sku_id': "user_cate_action_skus_" + suffix})
        user_cate_action1_brands = tmp[tmp.type == 1].groupby(['user_id', 'cate'])[
            'brand'].nunique().reset_index().rename(columns={'brand': "user_cate_action1_brands_" + suffix})
        user_cate_action2_brands = tmp[tmp.type == 2].groupby(['user_id', 'cate'])[
            'brand'].nunique().reset_index().rename(columns={'brand': "user_cate_action2_brands_" + suffix})
        user_cate_action3_brands = tmp[tmp.type == 3].groupby(['user_id', 'cate'])[
            'brand'].nunique().reset_index().rename(columns={'brand': "user_cate_action3_brands_" + suffix})
        user_cate_action4_brands = tmp[tmp.type == 4].groupby(['user_id', 'cate'])[
            'brand'].nunique().reset_index().rename(columns={'brand': "user_cate_action4_brands_" + suffix})
        user_cate_action_brands = tmp.groupby(['user_id', 'cate'])['brand'].nunique().reset_index().rename(
            columns={'brand': "user_cate_action_brands_" + suffix})
        user_cate_action_times = reduce(lambda x, y: pd.merge(x, y, on=['user_id', 'cate'], how='outer'), \
                                        [user_cate_action1_shops, user_cate_action2_shops, user_cate_action3_shops,
                                         user_cate_action4_shops, user_cate_action_shops, \
                                         user_cate_action1_skus, user_cate_action2_skus, user_cate_action3_skus,
                                         user_cate_action4_skus, user_cate_action_skus, \
                                         user_cate_action1_brands, user_cate_action2_brands, user_cate_action3_brands,
                                         user_cate_action4_brands, user_cate_action_brands])
        del user_cate_action1_shops, user_cate_action2_shops, user_cate_action3_shops, user_cate_action4_shops, user_cate_action_shops, \
            user_cate_action1_skus, user_cate_action2_skus, user_cate_action3_skus, user_cate_action4_skus, user_cate_action_skus, \
            user_cate_action1_brands, user_cate_action2_brands, user_cate_action3_brands, user_cate_action4_brands, user_cate_action_brands
        gc.collect()

        user_action_times['action_date'] = dat
        shop_action_times['action_date'] = dat
        cate_action_times['action_date'] = dat
        user_cate_action_times['action_date'] = dat
        return user_action_times, shop_action_times, cate_action_times, user_cate_action_times

    def add_user_behavier2_fea_day(self,sample,days):
        days_dict={
            '1day':["2018-04-08","2018-04-09","2018-04-10","2018-04-11","2018-04-12","2018-04-13","2018-04-14","2018-04-15"],
            '3day':["2018-04-06","2018-04-07","2018-04-08","2018-04-09","2018-04-10","2018-04-11","2018-04-12","2018-04-13"],
            '7day':["2018-04-02","2018-04-03","2018-04-04","2018-04-05","2018-04-06","2018-04-07","2018-04-08","2018-04-09"],
            '21day':["2018-03-19","2018-03-20","2018-03-21","2018-03-22","2018-03-23","2018-03-24","2018-03-25","2018-03-26"]
        }
        user_action_times0408, shop_action_times0408, cate_action_times0408, user_cate_action_times0408 = self.get_user_behavier_fea2(
            "2018-04-08", days_dict[days][0], days)
        user_action_times0409, shop_action_times0409, cate_action_times0409, user_cate_action_times0409 = self.get_user_behavier_fea2(
            "2018-04-09", days_dict[days][1], days)
        user_action_times0410, shop_action_times0410, cate_action_times0410, user_cate_action_times0410 = self.get_user_behavier_fea2(
            "2018-04-10", days_dict[days][2], days)
        user_action_times0411, shop_action_times0411, cate_action_times0411, user_cate_action_times0411 = self.get_user_behavier_fea2(
            "2018-04-11", days_dict[days][3], days)
        user_action_times0412, shop_action_times0412, cate_action_times0412, user_cate_action_times0412 = self.get_user_behavier_fea2(
            "2018-04-12", days_dict[days][4], days)

        user_action_times0413, shop_action_times0413, cate_action_times0413, user_cate_action_times0413 = self.get_user_behavier_fea2(
            "2018-04-13", days_dict[days][5], days)
        user_action_times0414, shop_action_times0414, cate_action_times0414, user_cate_action_times0414 = self.get_user_behavier_fea2(
            "2018-04-14", days_dict[days][6], days)
        user_action_times0415, shop_action_times0415, cate_action_times0415, user_cate_action_times0415 = self.get_user_behavier_fea2(
            "2018-04-15", days_dict[days][7], days)

        user_action_times = pd.concat([user_action_times0408, user_action_times0409, user_action_times0410, user_action_times0411,user_action_times0412, user_action_times0413, user_action_times0414,user_action_times0415])
        shop_action_times = pd.concat([shop_action_times0408, shop_action_times0409, shop_action_times0410, shop_action_times0411,shop_action_times0412, shop_action_times0413, shop_action_times0414,shop_action_times0415])
        cate_action_times = pd.concat([cate_action_times0408, cate_action_times0409, cate_action_times0410, cate_action_times0411,cate_action_times0412, cate_action_times0413, cate_action_times0414,cate_action_times0415])
        user_cate_action_times = pd.concat([user_cate_action_times0408, user_cate_action_times0409, user_cate_action_times0410,user_cate_action_times0411, user_cate_action_times0412, user_cate_action_times0413,user_cate_action_times0414, user_cate_action_times0415])

        sample = pd.merge(sample, user_action_times, on=['user_id', 'action_date'])
        sample = pd.merge(sample, shop_action_times, on=['shop_id', 'action_date'])
        sample = pd.merge(sample, cate_action_times, on=['cate', 'action_date'])
        sample = pd.merge(sample, user_cate_action_times, on=['user_id', 'cate', 'action_date'])
        return sample

    def add_user_behavier2_fea(self, sample):
        for days in ['1day','3day','7day','21day']:
            sample = self.add_user_behavier2_fea_day(sample,days)
            print('user behavier 2' + days + ' finished')
        return sample