#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""

import argparse
import pandas as pd
import os
from catboost import CatBoostRanker, Pool
from timeit import default_timer as timer
from tqdm import tqdm


def parse_mslr_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            target = int(parts[0])  
            QueryId = int(parts[1].split(":")[1])  
            
            features = {int(feat.split(":")[0]): float(feat.split(":")[1]) for feat in parts[2:]}
            
            row = {'target': target, 'QueryId': QueryId, **features}
            data.append(row)
    
    
    return pd.DataFrame(data)

def prepare_submission(dataset, scores, output_path):

        dataset['Score'] = scores
    
        dataset = dataset.sort_values(by=['QueryId', 'Score'], ascending=[True, False])

        dataset = dataset[['QueryId']]

        dataset['DocumentId'] = dataset.index

        dataset.to_csv(output_path, index=False)

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
        
        data_train = parse_mslr_file(os.path.join(args.data_dir, "Fold1", 'train.txt'))
        data_vali = parse_mslr_file(os.path.join(args.data_dir, "Fold1", "vali.txt"))
        X_train =  data_train.drop(columns=['target', 'QueryId']).to_numpy()
        Y_train = data_train['target'].to_numpy()
        Q_train = data_train['QueryId'].to_numpy().astype('uint32')
        X_val = data_vali.drop(columns=['target', 'QueryId']).to_numpy()
        Y_val = data_vali['target'].to_numpy()
        Q_val = data_vali['QueryId'].to_numpy().astype('uint32')
        
        train_pool = Pool(data=X_train, label=Y_train, group_id=Q_train)
        val_pool = Pool(data=X_val, label=Y_val, group_id=Q_val)
        
        model = CatBoostRanker(
            iterations=1000,                
            learning_rate=0.1,              
            depth=6,                        
            l2_leaf_reg=3,                  
            loss_function='YetiRank',        
            eval_metric='NDCG',              
            random_seed=42,                  
            early_stopping_rounds=50,        
            task_type='CPU',                 
            verbose=100,                    
        )
        model.fit(train_pool, eval_set=val_pool)
        model.save_model(args.model_file)
        print(f"Model saved")

    else:
        # Тут вы должны:
        # - загрузить тестовую часть датасета VKLR из папки args.data_dir
        # - загрузить модель из файла args.model_file
        # - применить модель ко всем запросам и документам из test.txt
        # - переранжировать документы по каждому из запросов так, чтобы более релевантные шли сначала
        # - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file
        data_test = parse_mslr_file(os.path.join(args.data_dir, "Fold1", 'test.txt'))
        X_test =  data_test.drop(columns=['target', 'QueryId']).to_numpy()
        Y_test = data_test['target'].to_numpy()
        Q_test = data_test['QueryId'].to_numpy().astype('uint32')
       
        test_pool = Pool(data=X_test, label=Y_test, group_id=Q_test)
   
        model = CatBoostRanker()
        model.load_model(args.model_file)
        scores = model.predict(test_pool)
        
        prepare_submission(data_test, scores, args.submission_file)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
