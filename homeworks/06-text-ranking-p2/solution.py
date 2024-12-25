#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution"""

import argparse
import math
import os
import pickle
import re
import string
from functools import lru_cache
from timeit import default_timer as timer

import nltk
import numpy as np
import pandas as pd
import pymorphy3
import tqdm
from nltk import tokenize
from nltk.corpus import stopwords
from pandarallel import pandarallel
from scipy.sparse import csc_matrix, csr_matrix

k1 = 1.5
b = 0.75


def bm25score(q, common, rows):
	indices = (q > 0).nonzero()[1]
	common = common[rows, :]
	res = np.sum(common[:, indices], axis=1)
	return res


def count_mean_tokens(total_documents, disks, args):
	mean_title = 0
	mean_body = 0
	for j in range(disks):
		with open(args.data_dir + f"/trans{CHUNKSIZE}trunc/trans_tfid{j}", 'rb') as file:
			title, body = pickle.load(file)
		mean_title += np.sum(title > 0)
		mean_body += np.sum(body > 0)
	mean_title /= total_documents
	mean_body /= total_documents
	return mean_title, mean_body


nltk.download('stopwords')


@lru_cache(maxsize=None)
def preprocess_word(morph, word):
	parsed = morph.parse(word)
	if not parsed:
		return None
	lemma = parsed[0].normal_form
	return lemma


# https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
class MyTfidfVectorizer:
	def __init__(self, data_dir):
		self.morph = pymorphy3.MorphAnalyzer(lang='ru')
		russian_sw = stopwords.words('russian')
		self.punc = string.punctuation
		self.russian_sw = set([self.morph.parse(word)[0].normal_form for word in russian_sw])
		self.words_count = []
		self.words_index = {}
		self.shape = None
		self.data_dir = data_dir
		self.pattern = re.compile(f"[{string.punctuation}]+")

	@lru_cache(maxsize=None)
	def is_good_word(self, word):
		return word is not None and word not in self.russian_sw and not re.fullmatch(self.pattern, word)

	def tokenize_func(self, text):
		tokens = tokenize.word_tokenize(text, "russian")
		filtered_tokens = []
		for word in tokens:
			word = word.lower()
			word = preprocess_word(self.morph, word)
			if self.is_good_word(word):
				filtered_tokens.append(word)
		return filtered_tokens

	def tokenize_batch(self, texts):
		result = texts.apply(self.tokenize_func).tolist()
		return result

	def vectorize(self, texts, tokenized=None):
		data = []
		rows = []
		cols = []
		if tokenized is None:
			tokenized = self.tokenize_batch(texts)
		for ind, text in enumerate(tokenized):
			words_frequency = {}
			for word in text:
				if word not in self.words_index:
					self.words_index[word] = len(self.words_index.keys())
				index = self.words_index[word]
				if index not in words_frequency:
					if index < len(self.words_count):
						self.words_count[index] += 1
					else:
						self.words_count.append(1)
				words_frequency[index] = words_frequency.setdefault(index, 0) + 1
			words_freq = np.asarray(list(words_frequency.values()))
			words_ind = np.asarray(list(words_frequency.keys()))
			data.append(words_freq)
			rows.append(ind * np.ones(len(words_frequency)))
			cols.append(words_ind)
		rows = np.concatenate(rows)
		data = np.concatenate(data)
		cols = np.concatenate(cols)
		return (data, (rows, cols)), tokenized

	def transform(self, texts):
		data = []
		rows = []
		cols = []
		tokenized_texts = texts.apply(lambda text: tokenize.word_tokenize(text, "russian")).tolist()
		for ind, tokens in enumerate(tokenized_texts):
			words_frequency = {}
			for word in tokens:
				word = word.lower()
				word = preprocess_word(self.morph, word)
				if word in self.words_index:
					index = self.words_index[word]
					words_frequency[index] = words_frequency.setdefault(index, 0) + 1
			words_freq = np.asarray(list(words_frequency.values()))
			words_ind = np.asarray(list(words_frequency.keys()))
			data.append(words_freq)
			rows.append(ind * np.ones(len(words_frequency)))
			cols.append(words_ind)
		rows = np.concatenate(rows)
		data = np.concatenate(data)
		cols = np.concatenate(cols)
		return csc_matrix((data, (rows, cols)), shape=(len(texts), len(self.words_index.keys())))


def fit_chunks(args, save_tokens=True, all_docs=None, disk_docs=None):
	vectorizer = MyTfidfVectorizer(args.data_dir)
	dataset_chunks = pd.read_csv(args.data_dir + "/vkmarco-docs.tsv", sep='\t', header=None, chunksize=CHUNKSIZE)
	folder_name = f"trans{CHUNKSIZE}"
	tokenize_name = f"/tokenize{CHUNKSIZE}"
	os.makedirs(os.path.join(args.data_dir, folder_name), exist_ok=True)
	used_disks = len(disk_docs.keys())
	if save_tokens:
		os.makedirs(args.data_dir + tokenize_name, exist_ok=True)
	progress = tqdm.tqdm(total=len(disk_docs.keys()))
	ind = 0
	for dataset in dataset_chunks:
		dataset[0] = dataset[0].str[1:].astype(int) - 1
		dataset = dataset.loc[dataset[0].isin(all_docs), :]
		if dataset.shape[0] == 0:
			continue
		dataset[2] = dataset[2].fillna("")
		dataset[3] = dataset[3].fillna("")
		tokens_path = args.data_dir + f"{tokenize_name}/tokens{ind}"
		if os.path.exists(tokens_path) and save_tokens == False:
			with open(tokens_path, 'rb') as file:
				tokenized_title, tokenized_body = pickle.load(file)
		else:
			tokenized_title, tokenized_body = None, None
		res_title, tokenized_title = vectorizer.vectorize(dataset[2], tokenized_title)
		res_body, tokenized_body = vectorizer.vectorize(dataset[3], tokenized_body)
		with open(os.path.join(args.data_dir, f"{folder_name}/trans_tfid{ind}"), 'wb') as file:
			pickle.dump((res_title, res_body), file)
		if save_tokens:
			with open(tokens_path, 'wb') as file:
				pickle.dump((tokenized_title, tokenized_body), file)
		ind += 1
		progress.update()
	vectorizer.words_count = np.asarray(vectorizer.words_count)
	with open(args.data_dir + f"/fitted_tfid{CHUNKSIZE}", 'wb') as file:
		pickle.dump(vectorizer, file)
	with open(args.data_dir + f"/fitted_tfid{CHUNKSIZE}", 'rb') as file:
		vectorizer = pickle.load(file)
	indexes2 = vectorizer.words_count > 4
	indexes2 = np.where(indexes2)[0]
	vec_shape = vectorizer.words_count.shape[0]
	vectorizer.words_count = vectorizer.words_count[indexes2]
	os.makedirs(f"{folder_name}trunc", exist_ok=True)
	progress = tqdm.tqdm(total=used_disks)
	for j in range(used_disks):
		with open(os.path.join(args.data_dir, f"{folder_name}/trans_tfid{j}"), 'rb') as file:
			res_title, res_body = pickle.load(file)
		title = res_to_sparse(res_title, indexes2, vec_shape)
		body = res_to_sparse(res_body, indexes2, vec_shape)
		with open(os.path.join(args.data_dir, f"{folder_name}trunc/trans_tfid{j}"), 'wb') as file:
			pickle.dump((title, body), file)
		progress.update()
	vectorizer.words_count = np.log(((all_docs.shape[0] - vectorizer.words_count + 0.5) /
	                                 (vectorizer.words_count + 0.5)) + 1.5)
	new = {}
	progres = tqdm.tqdm(total=len(indexes2))
	indexes2set = set(indexes2)
	cnt = 0
	for i, word in enumerate(vectorizer.words_index.keys()):
		if i in indexes2set:
			new[word] = cnt
			cnt += 1
			progres.update()
	vectorizer.words_index = new
	with open(args.data_dir + f"/fitted_tfid2", 'wb') as file:
		pickle.dump(vectorizer, file)
	with open(args.data_dir + f"/fitted_tfid2", 'rb') as file:
		vectorizer = pickle.load(file)
	return vectorizer


FILE = "dev"


def get_marks(args):
	queries_file = args.data_dir + f"/vkmarco-doc{FILE}-queries.tsv"
	queries_df = pd.read_csv(queries_file, sep='\t', header=None)
	queries_df.columns = ['query_id', 'query_text']

	qrels_file = args.data_dir + f"/vkmarco-doc{FILE}-qrels.tsv"
	qrels_df = pd.read_csv(qrels_file, sep=' ', header=None)
	qrels_df.columns = ['query_id', 'unused', 'docid', 'label']
	query_id2text = {query_id: query_text for query_id, query_text in
	                 zip(queries_df['query_id'], queries_df['query_text'])}

	queries_ids = dict(zip(list(queries_df["query_id"]), np.arange(queries_df.shape[0])))
	qrels = {}
	for i in range(0, len(qrels_df)):
		qrels_row = qrels_df.iloc[i]
		query_id = qrels_row['query_id']
		docid = int(qrels_row['docid'][1:])
		label = qrels_row['label']
		if label < 0 or label > 3:
			raise Exception(f"invalid label in qrels: doc_id = {docid}")
		docid2label = qrels.get(query_id)
		if docid2label is None:
			docid2label = {}
			qrels[query_id] = docid2label
		docid2label[docid] = label
	sample_sub = qrels_df.drop(columns=['unused', 'label'])
	sample_sub["docid"] = sample_sub["docid"].str[1:].astype(int) - 1
	all_docs = sample_sub["docid"].unique()
	sample_sub["disk"] = sample_sub["docid"] // CHUNKSIZE
	disk_docs = sample_sub.groupby("disk")["docid"].apply(lambda x: np.sort(pd.unique(x))).to_dict()
	disk_queries_docs = (sample_sub.groupby(["disk"])
	                     .apply(lambda d: d.groupby("query_id")["docid"].apply(list).to_dict(),
	                            include_groups=False)
	                     .to_dict())

	return query_id2text, qrels, disk_queries_docs, disk_docs, all_docs, queries_ids


def dcg(y, k=10):
	"""Computes DCG@k for a single query.

	y is a list of relevance grades sorted by position.
	len(y) could be <= k.
	"""
	r = 0.
	for i, y_i in enumerate(y):
		p = i + 1  # position starts from 1
		r += (2 ** y_i - 1) / math.log(1 + p, 2)
		if p == k:
			break
	return r


def ndcg(y, k=10):
	"""Computes NDCG@k for a single query.

	y is a list of relevance grades sorted by position.
	len(y) could be <= k.
	"""
	if len(y) == 0:
		return 0

	# Numerator
	dcg_k = dcg(y, k=k)
	# Denominator
	max_dcg = dcg(sorted(y, reverse=True), k=k)

	# Special case of all zeroes
	if max_dcg == 0:
		return 1.

	return dcg_k / max_dcg


CHUNKSIZE = 100


def res_to_sparse(res, indexes_leave, shape):
	data, (rows, cols) = res
	csc = csc_matrix((data, (rows, cols)), shape=(CHUNKSIZE, shape))[:, indexes_leave]
	result = csc.tocsr()
	return result


TOTAL_DOCS = 1258656

pandarallel.initialize(progress_bar=False)


def frames_eval(args):
	queries = pd.read_csv(args.data_dir + "/vkmarco-doceval-queries.tsv", sep='\t', header=None)
	queries_ids = dict(zip(list(queries[0]), np.arange(queries.shape[0])))
	query_id2text = {query_id: query_text for query_id, query_text in
	                 zip(queries[0], queries[1])}
	sample_sub = pd.read_csv(args.data_dir + "/sample_submission.csv")
	sample_sub["DocumentId"] = sample_sub["DocumentId"].str[1:].astype(int) - 1
	sample_sub["disk"] = sample_sub["DocumentId"] // CHUNKSIZE

	all_docs = sample_sub["DocumentId"].unique()
	queries_docs = sample_sub.groupby("QueryId")["DocumentId"].apply(list).to_dict()
	disk_docs = sample_sub.groupby("disk")["DocumentId"].apply(lambda x: np.sort(pd.unique(x))).to_dict()
	disk_queries_docs = (sample_sub.groupby(["disk"])
	                     .apply(lambda d: d.groupby("QueryId")["DocumentId"].apply(list).to_dict(),
	                            include_groups=False)
	                     .to_dict())
	return query_id2text, queries_ids, all_docs, queries_docs, disk_docs, disk_queries_docs


def main():
	parser = argparse.ArgumentParser(description='Text ranking homework solution')
	parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
	parser.add_argument('data_dir', help='input data directory')
	args = parser.parse_args()

	start = timer()
	np.random.seed(42)
	queries = pd.read_csv(args.data_dir + "/vkmarco-doceval-queries.tsv", sep='\t', header=None)
	query_id2text, queries_ids, all_docs, queries_docs, disk_docs, disk_queries_docs = frames_eval(args)
	vectorizer = fit_chunks(args, save_tokens=True, all_docs=all_docs, disk_docs=disk_docs)

	query_vec = vectorizer.transform(queries[1])

	queries_docs_scores = {key: [] for key in queries_docs.keys()}
	# mean_title, mean_body = count_mean_tokens(all_docs.shape[0], len(disk_docs.keys()), args)
	mean_title, mean_body = 6.955093179178545, 328.056831380002
	progress_bar = tqdm.tqdm(total=len(disk_docs.keys()))
	g = 0.75
	idf = csr_matrix(np.repeat(vectorizer.words_count[None, :], CHUNKSIZE, axis=0))
	for disk_ind, disk in enumerate(disk_docs.keys()):
		with open(args.data_dir + f"/trans{CHUNKSIZE}trunc/trans_tfid{disk_ind}", 'rb') as file:
			title, body = pickle.load(file)
		common_title = count_bm25(mean_title, title, idf)
		common_body = count_bm25(mean_body, body, idf)
		disk_queries = disk_queries_docs[disk].keys()
		for qId in disk_queries:
			docs_by_query = np.array(disk_queries_docs[disk][qId])
			docs_by_disk = disk_docs[disk]
			indexes = np.where(np.isin(disk_docs[disk], docs_by_query))[0]
			score_title = np.asarray(bm25score(query_vec[queries_ids[qId]], common_title, indexes))
			score_body = np.asarray(bm25score(query_vec[queries_ids[qId]], common_body, indexes))
			scores = score_title * g + (1 - g) * score_body
			queries_docs_scores[qId] += zip(scores.flatten().tolist(),
			                         docs_by_disk[np.isin(disk_docs[disk], docs_by_query)] + 1)
		progress_bar.update()

	with open(args.data_dir + "/heap.pkl", 'wb') as file:
		pickle.dump(queries_docs_scores, file)
	with open(args.data_dir + "/heap.pkl", 'rb') as file:
		queries_docs_scores = pickle.load(file)
	results = []
	for i, queries_scores in queries_docs_scores.items():
		df = pd.DataFrame(np.zeros((len(queries_scores), 2)), columns=["QueryId", "DocumentId"], dtype=str)
		queries_scores = sorted(queries_scores, key=lambda x: x[0], reverse=True)
		scores, indexes = zip(*queries_scores)
		indexes = np.array([f"D{int(n):09}" for n in indexes], dtype=str)
		df.iloc[:, 0] = (i * np.ones(len(queries_scores))).astype(int)
		df.iloc[:, 1] = indexes
		results.append(df)
	result = pd.concat(results, ignore_index=True)
	result.to_csv(args.submission_file, index=False)

	# Репортим время работы скрипта
	elapsed = timer() - start
	print(f"finished, elapsed = {elapsed:.3f}")


def count_bm25(mean_zone, zone_tf, idf):
	score = idf.multiply(zone_tf * (k1 + 1) /
	                     (zone_tf +
	                      k1 * (1 - b + b *
	                            (np.sum(zone_tf > 0, axis=1) / mean_zone))
	                      ))
	return score


if __name__ == "__main__":
	main()
