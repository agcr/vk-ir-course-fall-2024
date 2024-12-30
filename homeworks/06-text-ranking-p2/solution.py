#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution
./solution.py --submission_file=submission.csv text-ranking-homework-vk-ir-fall-2024
"""

from timeit import default_timer as timer
import pandas as pd
import tqdm
import argparse
import re
from nltk.corpus import stopwords
import nltk
import math

class BM25:
    def __init__(self, k=1.2, b=0.75):
        self.b = b
        self.k = k
        self.token_to_doc_count = {} # для каждого токена хранит количество документов, которые его содержат
        self.docs_tokens, self.avg_doc_size, self.docs_count = None, None, None 
        self.token_count_in_one_doc = [] # список, где на i-ой позиции словарь из token -> сколько раз встретился в i-ом документе

    def fit(self, docs_tokens):
        """
        предпосчет всего, что нужно для быстрого подсчета score'а
        """
        self.docs_tokens = docs_tokens
        self.docs_count = len(docs_tokens)
        
        num_docs = 0
        self.avg_doc_size = 0
        for tokens in docs_tokens:
            self.avg_doc_size += len(tokens)
            self.token_count_in_one_doc.append({})
            num_docs += 1
            if num_docs % 10_000 == 0: # остлеживаем время 
                print(f"fited {num_docs} docs...", end='\r')
            for token in tokens:
                cnt = self.token_count_in_one_doc[-1].get(token, 0)
                if cnt == 0:
                    self.token_to_doc_count[token] = self.token_to_doc_count.get(token, 0) + 1
                self.token_count_in_one_doc[-1][token] = cnt + 1
        self.avg_doc_size /= self.docs_count

    def predict(self, query_tokens, docs_index=None):
        if docs_index is None:
            docs_index = [i for i in range(self.docs_count)]
            
        result = []
        for i in docs_index:
            doc_tokens = self.docs_tokens[i]
            total_sum = 0
            for query_token in query_tokens:
                n_token = self.token_to_doc_count.get(query_token, 0) # в скольки документа встречается токен
                idf_score = math.log((self.docs_count - n_token + 0.5)/(n_token + 0.5) + 1)
                
                token_in_docs_count = self.token_count_in_one_doc[i].get(query_token, 0)
                rhs = (token_in_docs_count * (self.k + 1)) / (token_in_docs_count + self.k * (1 - self.b + self.b * (len(doc_tokens) / self.avg_doc_size)))
                
                total_sum += idf_score * rhs
            result.append((total_sum, i))
            i += 1
        return result
    
    def predict_and_get_id_top_k(self, query_tokens, docs_index=None, top_k=10):
        return [id for value, id in sorted(self.predict(query_tokens, docs_index), reverse=True)[:top_k]]
        

nltk.download('stopwords') # скачаем stop-слова

STOPWORDS = set(stopwords.words('russian'))

def get_index_of_docs(docids, docs):
    """
    по docids (например [D000003456, D000003460]) и списку документов 
    возвращает индексы списка, в которых представлены docids
    """
    res = []
    for i in range(len(docs)):
        if docs[i]['docid'] in docids:
            res.append(i)
    return res


def tokenize_text(text, n_gram_len=3): 
    """
    Принимает строку
    Возвращает список токенов (n-граммы заданной длины)
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens =  [token for token in text.lower().split() if token not in STOPWORDS]
    # res = []
    # for token in tokens:
    #     token_ngrams = ngrams(token, n_gram_len)
    #     doc_ngrams = [''.join(ngram) for ngram in token_ngrams]
    #     res += doc_ngrams
    return tokens


def read_docs(docs_file):
    """
    Принимает путь до файла
    Возвращает список документов в виде токенов
    """
    docs = []
    num_docs = 0
    with open(docs_file, 'rt') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            docid, url, title, body = parts
            docs.append({'docid': docid,'url': url, 'tokens': tokenize_text(f"{title} {body}")})
            num_docs += 1
            if num_docs % 10_000 == 0:
                print(f"added {num_docs} docs...", end='\r')
    return docs


def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()
    print("start")
    docs = read_docs(f"{args.data_dir}/vkmarco-docs.tsv")
    print(f"readed {len(docs)} docs")
    bm25 = BM25()
    bm25.fit([doc['tokens'] for doc in docs]) # обучим формулу на всех документах
    print("Bm25 ready. Start predict")
    queries_test = pd.read_csv(f"{args.data_dir}/vkmarco-doceval-queries.tsv", sep='\t', header=None)
    queries_test.columns = ['query_id', 'query_text']
    query_id2text = {query_id: query_text for query_id, query_text in zip(queries_test['query_id'], queries_test['query_text'])}
        
    sample_subm = pd.read_csv(f"{args.data_dir}/sample_submission.csv")
    id2docs = {} # qid to relevant docs
    for index, row in sample_subm.iterrows(): # для каждого запроса достаем релевантные документы
        qid, docid = row['QueryId'], row['DocumentId']
        id2docs[qid] = id2docs.get(qid, [])
        id2docs[qid].append(docid)

    res_qids = []
    res_docids = []
    query_ids = sorted(query_id2text.keys())
    
    for qid in tqdm.tqdm(query_ids):
        query_text_tokens = tokenize_text(query_id2text[qid]) # препроцессим текст
        docs_index = get_index_of_docs(id2docs[qid], docs) # берем индексы только релевантных документов

        result_docs = bm25.predict_and_get_id_top_k(query_text_tokens, docs_index=docs_index, top_k=10) # достаем id в порядке возрастания по BM25 score   
        found_docids = [docs[i]['docid'] for i in result_docs] # найдем фактические id документов
        for docid in found_docids: # Запишем результаты
            res_qids.append(qid)
            res_docids.append(docid)

    df = pd.DataFrame(data={'QueryId': res_qids, 'DocumentId': res_docids})
    df.to_csv(args.submission_file, index=False)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
