#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
import glob
import os
import pickle
import shutil
from collections import defaultdict
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import tqdm
from nltk import tokenize


def preprocess(text):
	# Tokenize
	tokenizer = tokenize.RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)

	# Normalize
	return [token.lower() for token in tokens]


BATCH_LEN = 10

WORD_DICT_BATCHES = 50


def save_index(inverted_index, index_dir):
	list_dict = list(inverted_index.items())
	word_to_batch = {}

	folder_path = os.path.join(index_dir, "splitted_index")
	if os.path.exists(folder_path):
		shutil.rmtree(folder_path)
	os.makedirs(folder_path, exist_ok=True)

	print("Saving index")
	progress = tqdm.tqdm(total=len(list_dict) // BATCH_LEN)
	for i in range(0, len(list_dict), BATCH_LEN):
		batch_dict = dict(list_dict[i:i + BATCH_LEN])
		batch_keys = list(batch_dict.keys())
		for j in range(len(batch_keys)):
			word_to_batch[batch_keys[j]] = (i + 1) // BATCH_LEN
		with open(os.path.join(folder_path, f"batch{(i + 1) // BATCH_LEN}"), 'wb') as file:
			pickle.dump(batch_dict, file)
		progress.update()
	step = round(len(list_dict) / WORD_DICT_BATCHES)

	folder_path = os.path.join(index_dir, "word_to_batch")
	if os.path.exists(folder_path):
		shutil.rmtree(folder_path)
	os.makedirs(folder_path, exist_ok=True)

	batches_dict = list(word_to_batch.items())
	for j in range(0, len(inverted_index.keys()), step):
		with open(os.path.join(folder_path, f"word_to_batch{(j + 1) // step}.pkl"), 'wb') as file:
			pickle.dump(dict(batches_dict[j:j + step]), file)


def documents_by_word(word, folder_path, all_batches_dicts):
	index = None
	for word_to_index in all_batches_dicts:
		with open(word_to_index, "rb") as file:
			word_to_index = pickle.load(file)
		if word in word_to_index:
			index = word_to_index[word]
			break
	if index is None:
		return set()
	with open(os.path.join(folder_path, f"batch{index}"), 'rb') as file:
		index_dict = pickle.load(file)
	docs = index_dict.get(word, set())
	return docs


TOTAL_DOCS = 9025


def main():
	# Парсим опции командной строки
	parser = argparse.ArgumentParser(description='Indexing homework solution')
	parser.add_argument('--submission_file', help='output Kaggle submission file')
	parser.add_argument('--build_index', action='store_true', help='force reindexing')
	parser.add_argument('--index_dir', required=True, help='index directory')
	parser.add_argument('data_dir', help='input data directory')
	args = parser.parse_args()

	# Будем измерять время работы скрипта
	start = timer()

	# Какой у нас режим: построения индекса или генерации сабмишна?
	if args.build_index:
		# Тут вы должны:
		# - загрузить тексты документов из файла args.data_dir/vkmarco-docs.tsv
		# - проиндексировать эти тексты, причем для разбиения текстов на слова (термины) надо воспользоваться функцией preprocess()
		# - сохранить получивший обратный индекс в папку переданную через параметр args.index_dir
		documents = pd.read_csv(args.data_dir + "/vkmarco-docs.tsv", sep='\t', header=None)
		documents[2] = documents[2].fillna("")
		documents[3] = documents[3].fillna("")
		documents["content"] = documents[2].str.cat(documents[3], sep=" ")
		documents_np = documents.to_numpy()
		inverted_index = defaultdict(set)
		for i in range(documents_np.shape[0]):
			doc = documents_np[i, 4]
			tokens = preprocess(doc)
			for j in tokens:
				inverted_index[j].add(documents_np[i, 0])
		save_index(inverted_index, args.index_dir)

	else:
		# Тут вы должны:
		# - загрузить поисковые запросы из файла args.data_dir/vkmarco-doceval-queries.tsv
		# - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
		# - для разбиения текстов запросов на слова тоже используем функцию preprocess()
		# - сформировать ваш сабмишн, в котором для каждого объекта (пары запрос-документ) будет проставлена метка 1 (в документе есть все слова из запроса) или 0
		#
		# Для формирования сабмишна надо загрузить и использовать файлы args.data_dir/sample_submission.csv и args.data_dir/objects.csv
		folder_path = os.path.join(args.index_dir, "splitted_index")
		all_words_batches = glob.glob(os.path.join(args.index_dir, "word_to_batch", "*"))
		if not all_words_batches:
			print("No index")
			return
		objs_chunks = pd.read_csv(os.path.join(args.data_dir, "objects.csv"), chunksize=9025)
		print("Generating submission")
		queries = pd.read_csv(args.data_dir + "/vkmarco-doceval-queries.tsv", sep="\t", header=None, chunksize=10)
		progress = tqdm.tqdm(total=100)
		for ind, queries_batch in enumerate(queries):
			queries_np = queries_batch.to_numpy()
			for i in range(queries_np.shape[0]):
				objs_df = pd.DataFrame(np.zeros((TOTAL_DOCS, 2), dtype=int), columns=["ObjectId", "Label"])
				query = preprocess(queries_np[i, 1])
				docs = documents_by_word(query[0], folder_path, all_words_batches)
				for j in range(1, len(query)):
					docs = docs.intersection(documents_by_word(query[j], folder_path, all_words_batches))
				objects = next(objs_chunks)
				objs_df.iloc[:, 0] = objects.iloc[:, 0]
				objs_df.iloc[np.where(np.isin(objects["DocumentId"], np.asarray(list(docs))))[0], 1] = 1
				if ind == 0 and i == 0:
					objs_df.to_csv(args.submission_file, header=True, index=False)
				else:
					objs_df.to_csv(args.submission_file, mode='a', header=False,
					               index=False)
				del objs_df
				progress.update()

	# Репортим время работы скрипта
	elapsed = timer() - start
	print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
	main()
