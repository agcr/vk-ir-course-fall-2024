#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simplified Learning to Rank Solution"""

import argparse
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostRanker, Pool

DEFAULT_PARAMS = {
    'iterations': 1000,
    'early_stopping_rounds': 100,
    'learning_rate': 0.05,
    'loss_function': 'YetiRank',
    'eval_metric': 'NDCG:top=5;type=Exp',
    'random_seed': 42,
    'verbose': 10,
    'task_type': 'GPU'
}

def load_and_process_file(path):
    chunk_size = 20000
    total_lines = sum(1 for _ in open(path))

    processed_chunks = []
    with tqdm(total=total_lines, desc="Processing file", unit="lines") as pbar:
        for chunk in pd.read_csv(path, sep=' ', header=None, chunksize=chunk_size):
            chunk.columns = ['label', 'qid'] + [f'feature_{i}' for i in range(1, len(chunk.columns) - 1)]
            
            chunk['qid'] = chunk['qid'].str.split(':', n=1).str[1].astype(int)
            
            feature_columns = chunk.columns[2:]
            chunk[feature_columns] = chunk[feature_columns].map(lambda x: float(x.split(':', 1)[1]))
            
            processed_chunks.append(chunk)
            pbar.update(len(chunk))

    return pd.concat(processed_chunks, ignore_index=True)

def to_catboost_dataset(df):
    X = df.drop(columns=['label', 'qid']).values
    y = df['label'].astype(float).values
    qids = df['qid'].values.astype(np.uint32)
    return Pool(data=X, label=y, group_id=qids)

def create_model():
    return CatBoostRanker(**DEFAULT_PARAMS)

def train_model(train_file, vali_file, model_file):
    print("Loading and preparing training data...")
    train_data = load_and_process_file(train_file)
    vali_data = load_and_process_file(vali_file)

    print("Converting data to CatBoost pools...")
    train_pool = to_catboost_dataset(train_data)
    vali_pool = to_catboost_dataset(vali_data)

    print("Creating model...")
    model = create_model()

    print("Training model...")
    start = timer()
    model.fit(train_pool, eval_set=vali_pool, use_best_model=True)
    print(f"Training completed in {timer() - start:.2f} seconds.")

    print(f"Saving model to {model_file}...")
    model.save_model(model_file)

def predict_and_generate_submission(test_file, model_file, submission_file):
    print("Loading and preparing test data...")
    test_data = load_and_process_file(test_file)

    print("Loading model...")
    model = CatBoostRanker()
    model.load_model(model_file)

    print("Predicting relevance scores...")
    test_pool = to_catboost_dataset(test_data)
    predictions = model.predict(test_pool)

    print("Generating submission file...")
    test_data['score'] = predictions
    submission = test_data.sort_values(['qid', 'score'], ascending=[True, False])
    submission = submission.reset_index().rename(columns={'qid': 'QueryId', 'index': 'DocumentId'})
    submission['QueryId'] = submission['QueryId'].astype(int)
    submission[['QueryId', 'DocumentId']].to_csv(submission_file, index=False)

    print(f"Submission saved to {submission_file}")

def main():
    parser = argparse.ArgumentParser(description='Simplified Learning to Rank solution')
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('--model_file', required=True, help='Path to save/load the model')
    parser.add_argument('--submission_file', help='Path to save the submission file')
    parser.add_argument('data_dir', help='Directory with train/vali/test datasets')
    args = parser.parse_args()

    start = timer()

    if args.train:
        print("Training mode enabled.")
        train_file = os.path.join(args.data_dir, 'train.txt')
        vali_file = os.path.join(args.data_dir, 'vali.txt')
        train_model(train_file, vali_file, args.model_file)
    else:
        print("Prediction mode enabled.")
        if not args.submission_file:
            raise ValueError("--submission_file must be specified in prediction mode.")
        test_file = os.path.join(args.data_dir, 'test.txt')
        predict_and_generate_submission(test_file, args.model_file, args.submission_file)

    print(f"Total elapsed time: {timer() - start:.2f} seconds.")

if __name__ == "__main__":
    main()