#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""

import argparse
import copy
import os
from timeit import default_timer as timer
import pandas as pd
import catboost

def read_dataframe(args, filename):
    df = pd.read_csv(os.path.join(args.data_dir, filename), sep=' ', header=None)
    df.columns = ['target', 'id'] + [i for i in range(df.shape[1] - 2)]
    df['id'] = df['id'].apply(lambda x: int(x.split(':')[1]))
    return df


def to_catboost_dataset(df):
    y = df['target'].to_numpy()
    q = df['id'].to_numpy().astype('uint32')
    X = df.drop(columns=['target', 'id']).to_numpy()
    return X, y, q


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

    # Какой у нас режим: обучения модели или генерации сабмишна?
    if args.train:
        df_train = read_dataframe(args, 'train.txt')
        df_val = read_dataframe(args, 'vali.txt')

        X_train, y_train, q_train = to_catboost_dataset(df_train)
        X_val, y_val, q_val = to_catboost_dataset(df_val)

        pool_train = catboost.Pool(data=X_train, label=y_train, group_id=q_train)
        pool_val = catboost.Pool(data=X_val, label=y_val, group_id=q_val)

        EVAL_METRIC = 'NDCG:top=5;type=Exp'

        DEFAULT_PARAMS = {
            'iterations': 1000,
            'early_stopping_rounds': 100,
            'eval_metric': EVAL_METRIC,
            'random_seed': 42,
            'verbose': 10
        }

        def create_model(loss_function):
            params = copy.deepcopy(DEFAULT_PARAMS)
            # Temporary directory that is used by catboost to store additional information
            catboost_info_dir = f"/tmp/catboost_info.{loss_function.lower()}"
            params.update({
                'loss_function': loss_function,
                'train_dir': str(catboost_info_dir),
            })
            return catboost.CatBoost(params)

        model = create_model('RMSE')
        start = timer()
        model.fit(pool_train, eval_set=pool_val, use_best_model=True)
        elapsed = timer() - start
        print(f"Model fit: num_trees = {model.tree_count_} elapsed = {elapsed:.3f}")

        model.save_model(args.model_file)

        # Тут вы должны:
        # - загрузить датасет VKLR из папки args.data_dir
        # - обучить модель с использованием train- и dev-сплитов датасета
        # - при необходимости, подобрать гиперпараметры
        # - сохранить обученную модель в файле args.model_file
    else:
        model = catboost.CatBoost()
        model.load_model(args.model_file)

        df_test = read_dataframe(args, 'test.txt')

        X_test, _, q_test = to_catboost_dataset(df_test)
        pool_test = catboost.Pool(data=X_test, group_id=q_test)
        y_predicted = model.predict(pool_test)

        df_results = pd.DataFrame({
            'QueryId': q_test,
            'DocumentId': range(len(q_test)),
            'predicted': y_predicted
        })

        df_results = df_results.sort_values(by=['QueryId', 'predicted'], ascending=[True, False])
        df_results = df_results[['QueryId', 'DocumentId']]
        df_results.to_csv(args.submission_file, index=False)

        # Тут вы должны:
        # - загрузить тестовую часть датасета VKLR из папки args.data_dir
        # - загрузить модель из файла args.model_file
        # - применить модель ко всем запросам и документам из test.txt
        # - переранжировать документы по каждому из запросов так, чтобы более релевантные шли сначала
        # - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
