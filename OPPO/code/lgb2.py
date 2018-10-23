import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import json

train_data = pd.read_table('../input/oppo_round1_train_20180929.txt',
        names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, encoding='utf-8').astype(str)
val_data = pd.read_table('../input/oppo_round1_vali_20180929.txt',
        names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, encoding='utf-8').astype(str)
test_data = pd.read_table('../input/oppo_round1_test_A_20180929.txt',
        names=['prefix', 'query_prediction', 'title', 'tag'], header=None, encoding='utf-8').astype(str)
train_data = train_data[train_data['label'] != '音乐']
test_data['label'] = -1

train_size = train_data.shape[0]
train_data = pd.concat([train_data,val_data])
df = pd.concat([train_data,test_data])

del train_data, test_data

df['label'] = df['label'].astype(int)

items = ['prefix', 'title', 'tag']

for item in items:
    temp = df.groupby(item, as_index=False)['label'].agg({item+'_click': 'sum', item+'_count': 'count'})
    temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'])
    df = pd.merge(df, temp, on=item, how='left')


for i in range(len(items)):
    for j in range(i+1, len(items)):
        item_g = [items[i], items[j]]
        temp = df.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
        temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
        df = pd.merge(df, temp, on=item_g, how='left')


def p_t(p, t):
    if p == t:
        return 1
    else:
        return 0


df['pt'] = df.apply(lambda x: p_t(x['prefix'], x['title']), axis=1)

tag = pd.factorize(df.tag)[0]
df['tag'] = tag


def prediction_list(text):
    if pd.isna(text): return [0]
    dic = json.loads(text)
    l = []
    for key in dic.keys():
        l.append(float(dic[key]))
    if len(l) == 0:
        l = [0]
    return l


df['pred_list'] = df['query_prediction'].apply(prediction_list)

df['pred_max'] = df['pred_list'].apply(lambda x: max(x))
df['pred_min'] = df['pred_list'].apply(lambda x: min(x))
df['pred_mean'] = df['pred_list'].apply(lambda x: np.mean(x))
df['pred_var'] = df['pred_list'].apply(lambda x: np.var(x))
df['pred_median'] = df['pred_list'].apply(lambda x: np.median(x))
df['pred_count'] = df['pred_list'].apply(lambda x: len(x)-1)

df = df.drop(['prefix', 'query_prediction', 'title', 'tag', 'pred_list'], axis=1)


test_data_ = df[df.label == -1]
train_data_ = df[df.label != -1]
val_data_ = train_data_.loc[train_size:]
train_data_ = train_data_.loc[:train_size]


print('train beginning')

X = np.array(train_data_.drop(['label'], axis=1))
y = np.array(train_data_['label'])
val_X = np.array(val_data_.drop(['label'], axis=1))
val_y = np.array(val_data_['label'])

X_test_ = np.array(test_data_.drop(['label'], axis=1))


print('================================')
print(X.shape)
print(y.shape)
print('================================')


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'is_unbalance': True,
    'lambda_l1': 0.1
}


lgb_train = lgb.Dataset(X, y)
lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50,
                verbose_eval=50,
                )
print(f1_score(val_y, np.where(gbm.predict(val_X, num_iteration=gbm.best_iteration) > 0.37, 1,0)))

pred = gbm.predict(X_test_, num_iteration=gbm.best_iteration)


test_data_['label'] = pred
test_data_['label'] = test_data_['label'].apply(lambda x: round(x))

#test_data_['label'].to_csv('../submit/result.csv',index = False)