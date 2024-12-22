#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution
./solution.py --submission_file=submission.csv text-ranking-homework-vk-ir-fall-2024
"""

from timeit import default_timer as timer
import pandas as pd
from rank_bm25 import BM25Okapi
import tqdm
import argparse
import pymorphy3 as pymorphy
import re
from nltk.corpus import stopwords
from razdel import tokenize
import nltk
nltk.download('stopwords') # скачаем stop-слова

STOPWORDS = set(stopwords.words('russian'))
MORPH = pymorphy.MorphAnalyzer(lang='ru')

def get_best_word(word):
    return MORPH.parse(word)[0].normal_form

def tokenize_text(text): 
    """
    Принимает строку
    Возвращает список токенов
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text.lower()
    tokens = [token.text for token in tokenize(text)]
    return [get_best_word(token) for token in tokens if token not in STOPWORDS]

def get_best_k(values, k=10):
    """
    Принимает на вход вектор скоров. 
    Возвращает список id'шников с максимальным скором
    """
    res = [] # value, id
    for i in range(len(values)):
        res.append((values[i], i))
    res.sort(reverse=True)
    return [id for value, id in res[:k]]


def read_docs(docs_file):
    """
    Принимает путь до файла
    Возвращает словарь вида docid -> {url: "", title: "" , body: ""}
    """
    docs = {}
    num_docs = 0
    with open(docs_file, 'rt') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            docid, url, title, body = parts
            docs[docid] = {'url': url, 'title': title, 'body': body}
            num_docs += 1
            if num_docs % 100_000 == 0:
                print(f"added {num_docs} docs...", end='\r')
    return docs


def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()

    docs = read_docs(f"{args.data_dir}/vkmarco-docs.tsv")
    print(f"readed {len(docs)} docs")
    
    sample_subm = pd.read_csv(f"{args.data_dir}/sample_submission.csv")
    id2docs = {} # qid to relevant docs
    for index, row in sample_subm.iterrows():
        qid, docid = row['QueryId'], row['DocumentId']
        id2docs[qid] = id2docs.get(qid, [])
        id2docs[qid].append(docid)

    queries_test = pd.read_csv(f"{args.data_dir}/vkmarco-doceval-queries.tsv", sep='\t', header=None)
    queries_test.columns = ['query_id', 'query_text']
    query_id2text = {query_id: query_text for query_id, query_text in zip(queries_test['query_id'], queries_test['query_text'])}

    res_qids = []
    res_docids = []
    query_ids = sorted(query_id2text.keys())

    for qid in tqdm.tqdm(query_ids):
        docs_ids = id2docs[qid] # достанем релевантные docid
        query_docs = []
        for docid in docs_ids:
            query_docs.append(docs[docid]) # достанем наполнение документов (docid, url, body, title)

        bm25 = BM25Okapi([tokenize_text(doc['title']) + tokenize_text(doc['body']) for doc in query_docs]) # обучим BM25
        
        query_text_tokens = tokenize_text(query_id2text[qid]) # препроцессим текст
        bm25_scores = bm25.get_scores(query_text_tokens) # считаем скоры
        result_docs = get_best_k(bm25_scores, 10) # достаем id в порядке возрастания по BM25Okapi score
        
        found_docids = [docs_ids[i] for i in result_docs] # найдем фактические id документов
        for docid in found_docids: # Запишем результаты
            res_qids.append(qid)
            res_docids.append(docid)
            
    df = pd.DataFrame(data={'QueryId': res_qids, 'DocumentId': res_docids})
    df.to_csv(args.submission_file, index=False)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
