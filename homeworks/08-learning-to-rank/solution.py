#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""

import argparse
import copy
import json
import os
from timeit import default_timer as timer

import catboost
import numpy as np
import pandas as pd
from catboost import datasets, utils

EVAL_METRIC = 'NDCG:top=5;type=Exp'

DEFAULT_PARAMS = {
    'iterations': 60000,
    'early_stopping_rounds': 10000,
    'eval_metric': EVAL_METRIC,
    'random_seed': 52,
    'verbose': 10,
    'task_type': 'GPU'
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

    # qid 0:1 -> qid 1
    df['qid'] = df['qid'].apply(lambda x: int(x.split(':')[1]))

    # 0:1 -> feature_i 1
    for i in range(num_features):
        feature_col = f"feature_{i+1}"
        df[feature_col] = df[feature_col].apply(lambda x: float(x.split(':')[1]))

    return df

def load_or_create_dataframe(data_dir, input_filename, processed_filename):
    processed_filepath = os.path.join(data_dir, processed_filename)

    # Проверка, существует ли предварительно сохранённый CSV
    if os.path.exists(processed_filepath):
        print(f"Loading processed data from {processed_filepath}")
        return pd.read_csv(processed_filepath)

    # Если CSV не существует, загружаем и обрабатываем оригинальный файл
    print(f"Processing raw data from {input_filename}")
    df = read_dataframe(data_dir, input_filename)

    # Сохраняем обработанный DataFrame для ускорения последующих загрузок
    print(f"Saving processed data to {processed_filepath}")
    df.to_csv(processed_filepath, index=False)

    return df


def to_catboost_dataset(df):
    y = df['label'].to_numpy()
    q = df['qid'].to_numpy().astype('uint32')
    X = df.drop(columns=['label', 'qid']).to_numpy()
    return (X, y, q)


def create_model(loss_function):
    params = copy.deepcopy(DEFAULT_PARAMS)

    # Временная директория для хранения информации catboost во время train
    catboost_info_dir = f"/content/catboost_info.{loss_function.lower()}"

    params.update({
        'loss_function': loss_function,
        'train_dir': str(catboost_info_dir),
    })
    return catboost.CatBoost(params)


def compute_metrics(y_true, y_hat, q):
    # Список метрик для подсчета
    eval_metrics = ['NDCG:top=5;type=Exp', 'DCG:top=5;type=Exp', 'MAP:top=5']

    for eval_metric in eval_metrics:
        scores = utils.eval_metric(y_true, y_hat, eval_metric, group_id=q)
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
        df_train = load_or_create_dataframe(args.data_dir, 'train.txt', 'train_processed.csv')
        df_val = load_or_create_dataframe(args.data_dir, 'vali.txt', 'vali_processed.csv')

        X_train, y_train, q_train = to_catboost_dataset(df_train)
        X_val, y_val, q_val = to_catboost_dataset(df_val)
        pool_train = catboost.Pool(data=X_train, label=y_train, group_id=q_train, )
        pool_val = catboost.Pool(data=X_val, label=y_val, group_id=q_val)

        model = create_model('YetiRank')
        model.fit(pool_train, eval_set=pool_val, use_best_model=True)
        model.save_model(args.model_file)
        compute_metrics(y_val, model.predict(pool_val), q_val)
        print(f"Model fit: num_trees = {model.tree_count_}")
    else:
        # Тут вы должны:
        # - загрузить тестовую часть датасета VKLR из папки args.data_dir
        # - загрузить модель из файла args.model_file
        # - применить модель ко всем запросам и документам из test.txt
        # - переранжировать документы по каждому из запросов так, чтобы более релевантные шли сначала
        # - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file
        model = catboost.CatBoost()
        model.load_model(args.model_file)

        df_test = load_or_create_dataframe(args.data_dir, 'test.txt', 'test_processed.csv')
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
