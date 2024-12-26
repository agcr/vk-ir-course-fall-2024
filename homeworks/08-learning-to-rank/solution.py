#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""

import argparse
from timeit import default_timer as timer
import os
import pandas as pd
from catboost import Pool, CatBoostRanker
from tqdm import tqdm


def read_mslr_file(file_path):
    data_list = []
    
    with open(file_path, 'r') as file:
        for line in tqdm(file):
            parts = line.strip().split()
            
            target = int(parts[0])
            
            query_id = int(parts[1].split(':')[1])
            
            features = {int(feat.split(':')[0]): float(feat.split(':')[1]) for feat in parts[2:]}
            
            row_dict = {'target': target, 'QueryId': query_id, **features}
            data_list.append(row_dict)
    
    df = pd.DataFrame(data_list)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Learning to Rank homework solution')
    parser.add_argument('--train', action='store_true', help='run script in model training mode')
    parser.add_argument('--model_file', help='output ranking model file')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()

    if args.train:
        train = read_mslr_file(os.path.join(args.data_dir, "Fold1", "train.txt"))
        vali = read_mslr_file(os.path.join(args.data_dir, "Fold1", "vali.txt"))
        train_data = train.drop(columns=['target', 'QueryId']).to_numpy()
        train_label = train['target'].to_numpy()
        train_group_id = train['QueryId'].to_numpy().astype('uint32')

        vali_data = vali.drop(columns=['target', 'QueryId']).to_numpy()
        vali_label = vali['target'].to_numpy()
        vali_group_id = vali['QueryId'].to_numpy().astype('uint32')

        train_pool = Pool(data=train_data, label=train_label, group_id=train_group_id)
        vali_pool = Pool(data=vali_data, label=vali_label, group_id=vali_group_id)
        model = CatBoostRanker(
            depth=8,
            iterations=1200,
            learning_rate=0.05,
            l2_leaf_reg=4,
            loss_function='YetiRank',
            eval_metric='NDCG',
            random_seed=123,
            early_stopping_rounds=100,
            verbose=200,
            max_bin=512,              
            od_type="Iter",    
            random_strength=1.5,
            rsm=0.8,
            task_type='CPU'
        )
        model.fit(
            train_pool,
            eval_set=vali_pool,
            use_best_model=True,
            verbose=True
        )
        model.save_model(args.model_file)
        
    else:
        test = read_mslr_file(os.path.join(args.data_dir, "Fold1", "test.txt"))
        test_data = test.drop(columns=['target', 'QueryId']).to_numpy()
        test_label = test['target'].to_numpy()
        test_group_id = test['QueryId'].to_numpy().astype('uint32')

        test_pool = Pool(data=test_data, label=test_label, group_id=test_group_id)

        model = CatBoostRanker()
        model.load_model(args.model_file)

        test_predictions = model.predict(test_pool)
        test['predictions'] = test_predictions
        sorted_test = test.sort_values(by=['QueryId', 'predictions'], ascending=[True, False])
        sorted_test['DocumentId'] = sorted_test.index
        submission = sorted_test[['QueryId', 'DocumentId']]
        submission.to_csv(args.submission_file, index=False)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
