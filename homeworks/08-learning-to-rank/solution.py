#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
./solution.py --train --model_file=output.model learning-to-rank-homework-vk-ir-fall-2024
./solution.py --submission_file=submission.csv --model_file=output.model learning-to-rank-homework-vk-ir-fall-2024
"""
"""Learning to Rank homework solution"""

import argparse
import copy
from timeit import default_timer as timer
import pandas as pd
import catboost
from catboost import utils
import pickle
import numpy as np

EVAL_METRIC = 'NDCG:top=10;type=Exp'
DEFAULT_PARAMS = {
    'iterations': 3000,            # maximum possible number of trees
    'early_stopping_rounds': 100,  # stop if metric does not improve for N rounds
    'eval_metric': EVAL_METRIC,    # # metric used for early stopping
    'random_seed': 42,
    'verbose': 50
}

def create_model(loss_function):
    params = copy.deepcopy(DEFAULT_PARAMS)

    # Temporary directory that is used by catboost to store additional information
    catboost_info_dir = f"/tmp/cat_boost_info.{loss_function.lower()}"

    params.update({
        'loss_function': loss_function,
        'train_dir': str(catboost_info_dir),
    })
    return catboost.CatBoost(params)

def to_catboost_dataset(df):
    y = df['label'].to_numpy()
    q = df['qid'].to_numpy().astype('uint32')
    X = df.drop(columns=['label', 'qid'])
    return (X, y, q)

def parse_mslr_line(line):
    if '#' in line:
        line, _ = line.split('#', 1)
    parts = line.strip().split()
    relevance = int(parts[0])
    query_id = parts[1].split(':')[1]
    features = {int(feat.split(':')[0]): float(feat.split(':')[1]) for feat in parts[2:]}
    return relevance, query_id, features

def read_mslr_to_dataframe(filename):
    rows = []
    with open(filename, 'r') as file:
        for line in file:
            relevance, query_id, features = parse_mslr_line(line)
            row = {'label': relevance, 'qid': query_id, **features}
            rows.append(row)
    return pd.DataFrame(rows)

def generate_column_names(num_features):
    """Generates column names for LETOR-like datasets"""
    columns = ['label', 'qid']
    for i in range(num_features):
        column = f"feature_{i+1}"
        columns.append(column)
    return columns

def create_new_df(df, best_features):
    df2 = df[['qid', 'label']]
    for f1 in best_features:
        df2[f1] = df[f1]
        for f2 in best_features:
            if f1 == f2:
                continue
            df2[f"{f1}_plus_{f2}"] = df[f1] + df[f2]
            df2[f"{f1}_minus_{f2}"] = df[f1] - df[f2]
            df2[f"{f1}_times_{f2}"] = df[f1] * df[f2]
            df2[f"{f1}_divided_by_{f2}"] = df[f1] / df[f2].replace(0, np.nan)
    return df


def get_order(scores):
    start_order = []
    res = []
    for i in range(len(scores)):
        res.append(0)
        start_order.append((scores[i][0], scores[i][1]))
    
    start_order.sort(reverse=True)
    for i in range(len(start_order)):
        res[i] = start_order[i][1]
    return res

def get_total_res(scores, quids):
    res = []
    cur_quid = quids[0]
    cur_scores = []
    for i in range(quids.shape[0]):
        if quids[i] != cur_quid:
            res += get_order(cur_scores)
            cur_scores = []
            cur_quid = quids[i]
        cur_scores.append((scores[i], i))
    res += get_order(cur_scores)
    return res


def main():
    print("Start")
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Learning to Rank homework solution')
    parser.add_argument('--train', action='store_true', help='run script in model training mode')
    parser.add_argument('--model_file', help='output ranking model file')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Будем измерять время работы скрипта
    start = timer()

    # Тут вы, скорее всего, должны проинициализировать фиксированным seed'ом генератор случайных чисел для того, чтобы ваше решение было воспроизводимо.
    # Например (зависит от того, какие конкретно либы вы используете):
    #
    # random.seed(42)
    # np.random.seed(42)
    # и т.п.
    best_features_file_name = "best_features.pkl"
    columns = generate_column_names(num_features=573)
    if args.train:
        df_val = read_mslr_to_dataframe(f"{args.data_dir}/Fold1/vali.txt")
        df_train = read_mslr_to_dataframe(f"{args.data_dir}/Fold1/train.txt")
        df_train.columns = columns
        df_val.columns = columns
        
        X_train, y_train, q_train = to_catboost_dataset(df_train)
        X_val, y_val, q_val = to_catboost_dataset(df_val)

        pool_train = catboost.Pool(data=X_train, label=y_train, group_id=q_train)
        pool_val = catboost.Pool(data=X_val, label=y_val, group_id=q_val)
        print("data was preprocessing. Start train model")

        model = create_model('YetiRank')
        model.fit(pool_train, eval_set=pool_val, use_best_model=True)

        feature_importances = model.get_feature_importance(type='FeatureImportance', data=pool_train)
        feature_names = X_train.columns
        score_and_feature_name = []
        for feature, importance in zip(feature_names, feature_importances):
            score_and_feature_name.append((importance, feature))
        score_and_feature_name.sort(reverse=True)
        best_features = [name for score, name in score_and_feature_name[:30]]
        with open(best_features_file_name, 'wb') as file: 
            pickle.dump({'best_features': best_features}, file) 
  
        df_train2 = create_new_df(df_train, best_features)
        df_val2 = create_new_df(df_val, best_features)
    
        X_train2, y_train2, q_train2 = to_catboost_dataset(df_train2)
        X_val2, y_val2, q_val2 = to_catboost_dataset(df_val2)
        pool_train2 = catboost.Pool(data=X_train2, label=y_train2, group_id=q_train2)
        pool_val2 = catboost.Pool(data=X_val2, label=y_val2, group_id=q_val2)
        model2 = create_model('YetiRank')
        model2.fit(pool_train2, eval_set=pool_val2, use_best_model=True)
        model2.save_model(args.model_file)

    else:
        model = catboost.CatBoost()
        model.load_model(args.model_file)
        best_features = []
        with open(best_features_file_name, 'rb') as file: 
            best_features = pickle.load(file)['best_features']

        df_test = read_mslr_to_dataframe(f"{args.data_dir}/Fold1/test.txt")
        df_test.columns = columns
        df_test2 = create_new_df(df_test, best_features)

        X_test, y_test, q_test = to_catboost_dataset(df_test2)
        pool_test = catboost.Pool(data=X_test, label=y_test, group_id=q_test)
        y_hat_test = model.predict(pool_test)

        df = pd.DataFrame(data={'QueryId': q_test, 'DocumentId': get_total_res(y_hat_test, q_test)})
        df.to_csv(args.submission_file, index=False)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
