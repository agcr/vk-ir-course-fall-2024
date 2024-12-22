#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""

import argparse
import copy
import os
from timeit import default_timer as timer

import catboost
import numpy as np
import pandas as pd
from catboost import datasets, utils

EVAL_METRIC = 'NDCG:top=5;type=Exp'

DEFAULT_PARAMS = {
    'iterations': 1000,
    'early_stopping_rounds': 100,
    'eval_metric': EVAL_METRIC,
    'random_seed': 52,
    'verbose': 10
}

def generate_column_names(num_features):
    columns = ['label', 'qid']
    for i in range(num_features):
        column = f"feature_{i+1}"
        columns.append(column)
    return columns

def read_dataframe(data_dir, filename):
    df = pd.read_csv(os.path.join(data_dir, filename), sep=' ', header=None)
    num_features = df.shape[1] - 2
    df.columns = generate_column_names(num_features)
    df['qid'] = df['qid'].apply(lambda x: int(x.split(':')[1]))
    for i in range(num_features):
        feature_col = f"feature_{i+1}"
        df[feature_col] = df[feature_col].apply(lambda x: float(x.split(':')[1]))
    
    return df

def to_catboost_dataset(df):
    y = df['label'].to_numpy()
    q = df['qid'].to_numpy().astype('uint32')
    X = df.drop(columns=['label', 'qid']).to_numpy()
    return (X, y, q)

def create_model(loss_function):
    params = copy.deepcopy(DEFAULT_PARAMS)

    # Temporary directory that is used by catboost to store additional information
    catboost_info_dir = f"homeworks/08-learning-to-rank/catboost_info.{loss_function.lower()}"

    params.update({
        'loss_function': loss_function,
        'train_dir': str(catboost_info_dir),
    })
    return catboost.CatBoost(params)

def compute_metrics(y_true, y_hat, q):
    # List of metrics to evaluate
    eval_metrics = ['NDCG:top=10;type=Exp', 'DCG:top=10;type=Exp', 'MAP:top=10']
    
    for eval_metric in eval_metrics:
        scores = utils.eval_metric(y_true, y_hat, eval_metric, group_id=q)
    
        # Print scores
        print(f"metric = {eval_metric} score = {scores[0]:.3f}")


def main():
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

    # Какой у нас режим: обучения модели или генерации сабмишна?
    if args.train:
        # Тут вы должны:
        # - загрузить датасет VKLR из папки args.data_dir
        # - обучить модель с использованием train- и dev-сплитов датасета
        # - при необходимости, подобрать гиперпараметры
        # - сохранить обученную модель в файле args.model_file
        df_train = read_dataframe(args.data_dir, 'train.txt')
        df_val = read_dataframe(args.data_dir, 'vali.txt')
        X_train, y_train, q_train = to_catboost_dataset(df_train)
        X_val, y_val, q_val = to_catboost_dataset(df_val)

        pool_train = catboost.Pool(data=X_train, label=y_train, group_id=q_train)
        pool_val = catboost.Pool(data=X_val, label=y_val, group_id=q_val)

        model = create_model('YetiRank')

        start = timer()
        model.fit(pool_train, eval_set=pool_val, use_best_model=True)
        elapsed = timer() - start

        model.save_model(args.model_file)
        compute_metrics(y_val, model.predict(pool_val), q_val)
        print(f"Model fit: num_trees = {model.tree_count_}\nModel train time = {elapsed:.3f}")
    else:
        # Тут вы должны:
        # - загрузить тестовую часть датасета VKLR из папки args.data_dir
        # - загрузить модель из файла args.model_file
        # - применить модель ко всем запросам и документам из test.txt
        # - переранжировать документы по каждому из запросов так, чтобы более релевантные шли сначала
        # - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file
        model = catboost.CatBoost()
        model.load_model(args.model_file)
        df_test = read_dataframe(args.data_dir, 'test.txt')

        X_test, y_test, q_test = to_catboost_dataset(df_test)
        pool_test = catboost.Pool(data=X_test, group_id=q_test)
        y_pred = model.predict(pool_test)

        compute_metrics(y_test, y_pred, q_test)

        df_results = pd.DataFrame({
            'QueryId': q_test,
            'DocumentId': range(len(q_test)),
            'predicted': y_pred
        })

        df_results = df_results.sort_values(by=['QueryId', 'predicted'], ascending=[True, False])
        df_results = df_results[['QueryId', 'DocumentId']]
        df_results.to_csv(args.submission_file, index=False)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
