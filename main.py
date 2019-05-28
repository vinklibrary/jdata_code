# -*- coding: utf-8 -*-
from utils.datapreprocess import *

if __name__ == '__main__':
    jdata_process_class = jdata_process()

    samples = jdata_process_class.gen_samples()
    print('Step1. 生成样本sample完成 '+str(samples.shape))
    samples = jdata_process_class.add_static_fea(samples)
    print('Step2.1 加入静态特征完成 ' + str(samples.shape))

    samples = jdata_process_class.add_comment_fea(samples)
    print(str(samples.shape))
    samples = jdata_process_class.add_conversion1_fea(samples)
    print(str(samples.shape))
    samples = jdata_process_class.add_conversion2_fea(samples)
    print(str(samples.shape))
    samples = jdata_process_class.add_user_behavier_fea(samples)
    print(str(samples.shape))
    samples = jdata_process_class.add_user_behavier2_fea(samples)
    print(str(samples.shape))
    samples = jdata_process_class.add_static_action_fea(samples)
    print(str(samples.shape))
    samples.to_csv("./dataset/sample_tmp.csv",index=None)
    print("用户分群完成")
    #生成样本

    #