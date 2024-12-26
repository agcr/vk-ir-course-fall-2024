#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""
import os
import copy
import argparse
import random
import tracemalloc
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from timeit import default_timer as timer

PARAMS_CATBOOST = {
    'iterations': 700,
    'random_seed': 42,
    'verbose': 20,
    'early_stopping_rounds': 70,
    'eval_metric': 'NDCG:top=5;type=Exp'
}
FUNC_LOSS = 'RMSE'


def measure_memory_usage(func):
    """Measure memory usage of `func`"""

    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        peak_memory = tracemalloc.get_traced_memory()[1]
        peak_memory_mb = peak_memory / 1024 / 1024
        print(f"Максимальное потребление памяти в {func.__name__}: {peak_memory_mb:.2f} МБ")
        tracemalloc.stop()
        return result

    return wrapper


def df_to_numpy(data):
    target = data['target'].to_numpy()
    id = data['id'].to_numpy().astype(np.int32)
    data = data.drop(columns=['target', 'id']).to_numpy()
    return data, target, id


def train_save_model(X_train, y_train, q_train, X_val, y_val, q_val, model_file):
    pool_train = Pool(data=X_train, label=y_train, group_id=q_train)
    pool_val = Pool(data=X_val, label=y_val, group_id=q_val)

    params = copy.deepcopy(PARAMS_CATBOOST)
    catboost_info_dir = f"/tmp/catboost_info.{FUNC_LOSS.lower()}"
    params.update({'depth': 6, 'loss_function': FUNC_LOSS, 'train_dir': str(catboost_info_dir)})

    model = CatBoost(params)
    model.fit(pool_train, eval_set=pool_val, use_best_model=True)
    model.save_model(model_file)


def load_dataframe(filename):
    data = pd.read_csv(filename, sep=' ', header=None)
    data.columns = ['target', 'id'] + [str(i) for i in range(data.shape[1] - 2)]
    data['id'] = data['id'].apply(lambda x: int(x.split(':')[1]))
    count_of_str = data.shape[1] - 2
    for i in range(count_of_str):
        data[str(i)] = data[str(i)].apply(lambda x: float(x.split(':')[1]))
    return data


def get_result_dataset(args, model):
    X_test, target, query_test = df_to_numpy(load_dataframe(os.path.join(args.data_dir, 'test.txt')))
    test_pool = Pool(data=X_test, group_id=query_test)
    predictions = model.predict(test_pool)
    result = pd.DataFrame({'QueryId': query_test, 'DocumentId': range(len(query_test)), 'pred': predictions})
    result = result.sort_values(by=['QueryId', 'pred'], ascending=[True, False])[['QueryId', 'DocumentId']]
    result.to_csv(args.submission_file, index=False)


@measure_memory_usage
def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Learning to Rank homework solution')
    parser.add_argument('--train', action='store_true', help='run script in model training mode')
    parser.add_argument('--model_file', help='output ranking model file')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()
    start = timer()
    random.seed(42)
    if args.train:
        df_train = load_dataframe(os.path.join(args.data_dir, 'train.txt'))
        df_val = load_dataframe(os.path.join(args.data_dir, 'vali.txt'))
        X_train, y_train, query_train = df_to_numpy(df_train)
        X_val, y_val, query_val = df_to_numpy(df_val)
        train_save_model(X_train, y_train, query_train, X_val, y_val, query_val, args.model_file)
    else:
        model = CatBoost()
        model.load_model(args.model_file)
        get_result_dataset(args, model)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
