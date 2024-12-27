#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Learning to Rank homework solution"""

import argparse
import copy
import os.path
from timeit import default_timer as timer
from tqdm import tqdm
import catboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(file_path, name, save=False):
	dataset = []
	if os.path.exists(name):
		df = pd.read_csv(name)
		return df
	with open(file_path, "r") as file:
		for i, line in enumerate(file):
			parts = line.strip().split()
			data = [parts[0], parts[1].split(":")[1]]
			for features in parts[2:]:
				data.append(features.split(":")[1])
			dataset.append(data)
			if i > 200_000:
				break
	df = pd.DataFrame(dataset, columns=["rel", "qid"] + [f"f{i}" for i in range(2, len(dataset[0]))])
	if save:
		df.to_csv(name, index=False)
		df = pd.read_csv(name)
	return df


def to_catboost_dataset(df):
	y = df['rel'].to_numpy()
	q = df['qid'].to_numpy().astype('uint32')
	x = df.drop(columns=["rel", 'qid']).to_numpy()
	return x, y, q


def create_model(loss_function):
	params = copy.deepcopy(DEFAULT_PARAMS)
	catboost_info_dir = f"/tmp/catboost_info.{loss_function.lower()}"

	params.update({
		'loss_function': loss_function,
		'train_dir': str(catboost_info_dir),
	})
	return catboost.CatBoost(params)


EVAL_METRIC = 'NDCG:top=5;type=Exp'

DEFAULT_PARAMS = {
	'iterations': 1000,
	'early_stopping_rounds': 100,
	'depth': 6,
	'learning_rate': 0.08,
	'l2_leaf_reg': 3,
	'random_strength': 1,
	'loss_function': 'YetiRank',
	'eval_metric': EVAL_METRIC,
	'random_seed': 22,
	'verbose': 10
}


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
	np.random.seed(42)
	# и т.п.

	# Какой у нас режим: обучения модели или генерации сабмишна?
	if args.train:
		# Тут вы должны:
		# - загрузить датасет VKLR из папки args.data_dir
		# - обучить модель с использованием train- и dev-сплитов датасета
		# - при необходимости, подобрать гиперпараметры
		# - сохранить обученную модель в файле args.model_file
		data_vali = read_dataset(args.data_dir + "/vali.txt", args.data_dir + "/vali.csv", save=True)
		data_train, data_vali = train_test_split(data_vali, test_size=0.2, random_state=42)
		data_train.sort_values(by="qid", ascending=True, inplace=True)
		data_vali.sort_values(by="qid", ascending=True, inplace=True)
		X_vali, y_vali, q_vali = to_catboost_dataset(data_vali)
		X_train, y_train, q_train = to_catboost_dataset(data_train)
		pool_train = catboost.Pool(data=X_train, label=y_train, group_id=q_train)
		pool_vali = catboost.Pool(data=X_vali, label=y_vali, group_id=q_vali)

		model = create_model("YetiRank")
		model.fit(pool_train, eval_set=pool_vali, use_best_model=True)
		model.save_model(args.model_file)

	else:
		# Тут вы должны:
		# - загрузить тестовую часть датасета VKLR из папки args.data_dir
		# - загрузить модель из файла args.model_file
		# - применить модель ко всем запросам и документам из test.txt
		# - переранжировать документы по каждому из запросов так, чтобы более релевантные шли сначала
		# - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file
		data_test = read_dataset(args.data_dir + "/test.txt", args.data_dir + "/test.csv", save=True)
		X_test, y_test, q_test = to_catboost_dataset(data_test)
		pool_test = catboost.Pool(data=X_test, label=y_test, group_id=q_test)
		model = create_model("YetiRank")
		model.load_model(args.model_file)
		prediction = model.predict(pool_test)
		query_ids = data_test["qid"].unique().astype(int)
		submission = []
		ind = 0
		progress = tqdm(total=query_ids.shape[0])
		for query_id in query_ids:
			length = data_test.loc[data_test["qid"] == query_id].shape[0]
			docs_ranks = prediction[ind: ind + length]
			ranks = np.argsort(docs_ranks)[::-1]
			ranks += ind
			ind += length
			submission.append(np.stack([query_id * np.ones(length, dtype=int), ranks.astype(int)], axis=1))
			progress.update()
		df = pd.DataFrame(np.concatenate(submission), columns=["QueryId", "DocumentId"])
		df.to_csv(args.submission_file, index=False)

	# Репортим время работы скрипта
	elapsed = timer() - start
	print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
	main()
