#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""

import argparse
from timeit import default_timer as timer
import random
import numpy as np
import catboost
import pandas as pd

def set_seed(seed):
    """Устанавливаем фиксированный сид для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)

def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Learning to Rank homework solution')
    parser.add_argument('--train', action='store_true', help='run script in model training mode')
    parser.add_argument('--model_file', required=True, help='output ranking model file')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Устанавливаем фиксированный сид
    set_seed(42)

    # Будем измерять время работы скрипта
    start = timer()

    if args.train:
        # Режим обучения модели
        train_path = f"{args.data_dir}/train.txt"
        val_path = f"{args.data_dir}/vali.txt"

        # Загрузка данных
        def to_catboost_from_file(filepath):
            df = pd.read_csv(filepath, sep=' ', header=None)
            num_features = df.shape[1] - 2
            columns = ['label', 'qid'] + [f"feature_{i+1}" for i in range(num_features)]
            df.columns = columns
            df['qid'] = df['qid'].apply(lambda x: int(x.split(':')[1]))
            for col in columns[2:]:
                df[col] = df[col].apply(lambda x: float(x.split(':')[1]))
            y = df['label'].to_numpy()
            q = df['qid'].to_numpy().astype('uint32')
            X = df.drop(columns=['label', 'qid']).to_numpy()
            return X, y, q

        X_train, y_train, q_train = to_catboost_from_file(train_path)
        X_val, y_val, q_val = to_catboost_from_file(val_path)

        pool_train = catboost.Pool(X_train, label=y_train, group_id=q_train)
        pool_val = catboost.Pool(X_val, label=y_val, group_id=q_val)

        # Параметры модели
        model = catboost.CatBoost({
            'iterations': 1000,
            'early_stopping_rounds': 100,
            'eval_metric': 'NDCG:top=10;type=Exp',
            'loss_function': 'YetiRank',
            'random_seed': 42,
            'verbose': 10
        })

        # Обучение
        model.fit(pool_train, eval_set=pool_val, use_best_model=True)
        model.save_model(args.model_file)

    else:
        # Режим генерации сабмишна
        test_path = f"{args.data_dir}/test.txt"

        def to_catboost_test(filepath):
            df = pd.read_csv(filepath, sep=' ', header=None)
            num_features = df.shape[1] - 2
            columns = ['label', 'qid'] + [f"feature_{i+1}" for i in range(num_features)]
            df.columns = columns
            df['qid'] = df['qid'].apply(lambda x: int(x.split(':')[1]))
            for col in columns[2:]:
                df[col] = df[col].apply(lambda x: float(x.split(':')[1]))
            q = df['qid'].to_numpy().astype('uint32')
            X = df.drop(columns=['label', 'qid']).to_numpy()
            return X, q

        X_test, q_test = to_catboost_test(test_path)
        pool_test = catboost.Pool(X_test, group_id=q_test)

        # Загрузка модели
        model = catboost.CatBoost()
        model.load_model(args.model_file)

        # Прогноз и формирование сабмишна
        y_hat_test = model.predict(pool_test)
        df_results = pd.DataFrame({
            'QueryId': q_test,
            'DocumentId': range(len(q_test)),
            'PredictedRel': y_hat_test
        })
        df_results = df_results.sort_values(by=['QueryId', 'PredictedRel'], ascending=[True, False])
        df_results = df_results[['QueryId', 'DocumentId']]
        df_results.to_csv(args.submission_file, index=False)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
