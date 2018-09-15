# coding: utf-8
from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn import cross_validation,metrics
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from datetime import datetime, timedelta, timezone
import numpy as np
from tqdm import tqdm
import gc
import sys

stratified = True
seed = 0
n_splits = 5
num_boost_round = 10000
drop_zero_flag = True
drop_manual_flag = True
def load_datasets(feats):
    dfs = [pd.read_feather(f'../feather/{f}_train.ftr') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f'../feather/{f}_test.ftr') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test

def drop_zero(df):#importanceが0の特徴量を削除
    feature = pd.read_csv('../importance/del.csv').feature.values
    feature = list(set(df.columns) & set(feature))
    df = df.drop(feature,axis=1)
    return df

def drop_manual(df):#カラムを個別に削除
    feature = ['ORGANIZATION_TYPE']
    df = df.drop(feature,axis=1)
    return df

if __name__=="__main__":
    logger = getLogger(__name__)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler('../log/'+__file__+'.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.info('start')

    ss = pd.read_csv("../input/sample_submission.csv")
    feats = ['application','ext_mean','sum_null','sum_flag_document','sum_flag_contact','comb_ext','flag_own','sum_avg','region_city','money','weekend','comb_ext_fill','kernel','money_plus','type_suite','days_ratio','days_diff']
    table = ['pos_cash_balance_v2','my_pos_cash_balance_v2','bureau','my_bureau_v6','my_bureau_part2_v5','installments_payments','my_installments_payments','my_installments_payments_part2','my_installments_payments_part3','credit_card_balance','my_credit_card_balance','my_credit_card_balance_part2','my_previous_application','my_previous_application_part2_v4','my_previous_application_part3_v2']
    feats.extend(table)
    train_df, test_df = load_datasets(feats)

    if drop_zero_flag:
        train_df = drop_zero(train_df)
        test_df = drop_zero(test_df)

    if drop_manual_flag:
        train_df = drop_manual(train_df)
        test_df = drop_manual(test_df)
    train_df.head(100).to_csv('../output/latest_feature.csv', index= True)

    categorical_columns = []
    for column in train_df.columns:
        if train_df[column].dtype == "object":
            categorical_columns.append(column)

    for column in categorical_columns:
        train_df[column] = train_df[column].astype('category')
        test_df[column] = test_df[column].astype('category')

    if stratified:
        folds = StratifiedKFold(n_splits= n_splits, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= n_splits, shuffle=True, random_state=seed)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx],
                             label=train_df['TARGET'].iloc[train_idx],
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx],
                             label=train_df['TARGET'].iloc[valid_idx],
                             free_raw_data=False, silent=True)

        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,
            'num_leaves': 20,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.6,
            'min_split_gain': 0.0222415,
            'min_child_weight': 60,
            "min_data_in_leaf":20,
            'seed': 0,
            'verbose': -1,
            'metric': 'auc',
        }

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=200,
            verbose_eval=False
        )

        oof_preds[valid_idx] = clf.predict(dvalid.data)
        sub_preds += clf.predict(test_df[feats]) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        temp = roc_auc_score(dvalid.label, oof_preds[valid_idx])
        logger.info('Fold: {}, AUC: {}'.format(n_fold + 1, temp))
        del clf, dtrain, dvalid
        gc.collect()

    temp = roc_auc_score(train_df['TARGET'], oof_preds)
    logger.info('Full AUC score: {}'.format(temp))

    JST = timezone(timedelta(hours=+9), 'JST')
    file_name = datetime.now(JST).strftime('%y%m%d%H%M')+'.csv'
    sub_df = test_df[['SK_ID_CURR']].copy()
    sub_df['TARGET'] = sub_preds
    sub_df[['SK_ID_CURR', 'TARGET']].to_csv('../output/'+file_name, index= False)
    feature_importance_df_mean = feature_importance_df.groupby('feature').mean().drop('fold',axis=1).sort_values('importance', ascending=False)
    feature_importance_df_mean.to_csv('../importance/'+file_name, index= True)
    feature_importance_df_mean.to_csv('../importance/latest.csv', index= True)
    logger.info('end')
