#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""

import argparse
import copy
from timeit import default_timer as timer

import catboost
import numpy as np
import pandas as pd
import tqdm

DEFAULT_PARAMS = {
    'iterations': 70000,
    'early_stopping_rounds': 10000,
    'eval_metric': 'NDCG:top=10;type=Exp',
    'random_seed': 42,
    'verbose': 10,
    'task_type': 'GPU'
}


def create_model(loss_function, path):
    params = copy.deepcopy(DEFAULT_PARAMS)
    catboost_info_dir = f"/tmp/catboost_info.{loss_function.lower()}"
    params.update({
        'loss_function': loss_function,
        'train_dir': str(catboost_info_dir),
    })
    return catboost.CatBoost(params)


def to_catboost_dataset(df):
    y = df['label'].to_numpy()
    q = df['qid'].to_numpy().astype('uint32')
    X = df.drop(columns=['label', 'qid']).to_numpy()
    return (X, y, q)


def proceed_file(path):
    df = pd.DataFrame()
    total_lines = sum(1 for _ in open(path))
    chunk_size = 15000
    for chunk in tqdm.tqdm(pd.read_csv(path, sep=' ', header=None, chunksize=chunk_size),
                           total=total_lines // chunk_size,
                           desc="Processing file chunks"):
        chunk.columns = ['label', 'qid'] + [str(i) for i in range(1, 574)]
        chunk['qid'] = chunk['qid'].str.split(':').str[1].astype(int)
        for col in chunk.columns[2:]:
            chunk[col] = chunk[col].str.split(':').str[1].astype(float)
        df = pd.concat([df, chunk], ignore_index=True)
    return df


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
    np.random.seed(42)
    # Какой у нас режим: обучения модели или генерации сабмишна?
    if args.train:
        # Тут вы должны:
        # - загрузить датасет VKLR из папки args.data_dir
        # - обучить модель с использованием train- и dev-сплитов датасета
        # - при необходимости, подобрать гиперпараметры
        # - сохранить обученную модель в файле args.model_file
        df_train = proceed_file(args.data_dir + '/train.txt')
        df_vali = proceed_file(args.data_dir + '/vali.txt')
        
        X_train, y_train, q_train = to_catboost_dataset(df_train)
        X_vali, y_vali, q_vali = to_catboost_dataset(df_vali)

        pool_train = catboost.Pool(data=X_train, label=y_train, group_id=q_train)
        pool_vali = catboost.Pool(data=X_vali, label=y_vali, group_id=q_vali)

        model = create_model('YetiRank', args.data_dir)

        start_fit = timer()
        model.fit(pool_train, eval_set=pool_vali, use_best_model=True)
        elapsed = timer() - start_fit
        print(f"Model fit: num_trees = {model.tree_count_} elapsed = {elapsed:.3f}")

        model.save_model(args.model_file)
    else:
        # Тут вы должны:
        # - загрузить тестовую часть датасета VKLR из папки args.data_dir
        # - загрузить модель из файла args.model_file
        # - применить модель ко всем запросам и документам из test.txt
        # - переранжировать документы по каждому из запросов так, чтобы более релевантные шли сначала
        # - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file
        df_test = proceed_file(args.data_dir + '/test.txt')
        X_test, y_test, q_test = to_catboost_dataset(df_test)
        pool_test = catboost.Pool(data=X_test, label=y_test, group_id=q_test)

        model = catboost.CatBoost()
        model.load_model(args.model_file)

        y_test_labels = model.predict(pool_test)
        df_test['label'] = y_test_labels
        df_test['ind'] = df_test.index
        df_sorted = df_test.sort_values(by=['qid', 'label'], ascending=[True, False])

        res = df_sorted[['qid', 'ind']].rename(columns={'qid': 'QueryId', 'ind': 'DocumentId'})
        res.to_csv(args.submission_file, index=False)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
