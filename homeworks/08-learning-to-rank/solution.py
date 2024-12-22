#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""
import argparse
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from tqdm import tqdm


def load_data(file_path):
    data = pd.read_csv(file_path, sep=' ', header=None)
    data.columns = ['target', 'id'] + [f'feature_{i}' for i in range(data.shape[1] - 2)]

    data['id'] = data['id'].apply(lambda x: int(x.split(':')[1]))
    for col in tqdm(data.columns[2:]):
        data[col] = data[col].apply(lambda x: float(x.split(':')[1]))

    return data


def prepare_data(data):
    features = data.drop(columns=['target', 'id']).values
    target = data['target'].values
    group_ids = data['id'].astype(np.int32).values
    return features, target, group_ids


def train_and_save_model(train_features, train_labels, train_groups,
                         val_features, val_labels, val_groups, model_path):
    train_pool = Pool(data=train_features, label=train_labels, group_id=train_groups)
    val_pool = Pool(data=val_features, label=val_labels, group_id=val_groups)

    params = {
        'iterations': 500,
        'max_depth': 7,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': 50,
        'early_stopping_rounds': 80,
        'eval_metric': 'NDCG:top=5;type=Exp',
        'thread_count': -1,
    }

    model = CatBoost(params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    model.save_model(model_path)


def generate_submission(args, model):
    test_data = load_data(os.path.join(args.data_dir, 'test.txt'))
    test_features, _, test_groups = prepare_data(test_data)

    test_pool = Pool(data=test_features, group_id=test_groups)
    predictions = model.predict(test_pool)

    submission = pd.DataFrame({
        'QueryId': test_groups,
        'DocumentId': range(len(test_groups)),
        'Prediction': predictions
    }).sort_values(by=['QueryId', 'Prediction'], ascending=[True, False])

    submission[['QueryId', 'DocumentId']].to_csv(args.submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Learning to Rank homework solution')
    parser.add_argument('--train', action='store_true', help='run script in model training mode')
    parser.add_argument('--model_file', help='output ranking model file')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()

    if args.train:
        train_data = load_data(os.path.join(args.data_dir, 'train.txt'))
        val_data = load_data(os.path.join(args.data_dir, 'vali.txt'))

        train_features, train_labels, train_groups = prepare_data(train_data)
        val_features, val_labels, val_groups = prepare_data(val_data)

        train_and_save_model(train_features, train_labels, train_groups,
                             val_features, val_labels, val_groups, args.model_file)
    else:
        model = CatBoost()
        model.load_model(args.model_file)
        generate_submission(args, model)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
