import lightgbm as lgb

class lgb_model():
    def __init__(self):
        self.model = []
        self.params = {'nthread': 4,  # 进程数
                  'max_depth': 8,  # 最大深度
                  'learning_rate': 0.05,  # 学习率
                  'bagging_fraction': 0.95,  # 采样数
                  'num_leaves': 50,  # 终点节点最小样本占比的和
                  'feature_fraction': 0.2,  # 样本列采样
                  'objective': 'binary',
                  'lambda_l1': 10,  # L1 正则化
                  'lambda_l2': 2,  # L2 正则化
                  'bagging_seed': 100,  # 随机种子,light中默认为100
                  'verbose': 0,
                  'metric': 'auc'
                  }

    def train(self,sample,features):
        self.model = []
        train_data = lgb.Dataset(data=sample[features], label=sample.flag)
        self.model.append(lgb.train(self.params, train_data, num_boost_round=700))
        self.model.append(lgb.train(self.params, train_data, num_boost_round=1000))
        self.model.append(lgb.train(self.params, train_data, num_boost_round=1300))

    def predict(self,sample,features):
        if len(self.model) < 3:
            print("the model haven't train yet")
            return None
        predicts = []
        for i in range(3):
            predicts.append(self.model[i].predict(sample[features]))
        sample['predict'] = (predicts[0]+predicts[1]+predicts[2]/3)
        return sample