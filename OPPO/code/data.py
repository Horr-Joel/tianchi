import pandas as pd
train = pd.read_table('../input/oppo_round1_train_20180929.txt',
        names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, encoding='utf-8').astype(str)
val = pd.read_table('../input/oppo_round1_vali_20180929.txt',
        names=['prefix', 'query_prediction', 'title', 'tag', 'label'], header=None, encoding='utf-8').astype(str)
test = pd.read_table('../input/oppo_round1_test_A_20180929.txt',
        names=['prefix', 'query_prediction', 'title', 'tag'], header=None, encoding='utf-8').astype(str)

train = train[train.label != '音乐']
train.label = train.label.astype(int)
val.label = val.label.astype(int)


#总体点击率为0.372

# 0.8 输入部分   占比               大部分输入都是部分输入
train[train.prefix != train.title].shape[0]/train.shape[0]
val[val.prefix != val.title].shape[0]/val.shape[0]

# 0.29  输入部分并且点击的  占比        大部分点击都是输入部分就点
train[train.prefix != train.title].label.astype(int).sum()/train.shape[0]
val[val.prefix != val.title].label.astype(int).sum()/val.shape[0]


# 0.358  输入部分并点击   在输入部分的占比
train[train.prefix != train.title].label.astype(int).sum()/train[train.prefix != train.title].shape[0]
val[val.prefix != val.title].label.astype(int).sum()/val[val.prefix != val.title].shape[0]

# 0.43 pd 输入全称并点击  在输入全称的占比
train[train.prefix == train.title].label.astype(int).sum()/train[train.prefix == train.title].shape[0]
val[val.prefix == val.title].label.astype(int).sum()/val[val.prefix == val.title].shape[0]