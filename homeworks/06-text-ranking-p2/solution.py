#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution"""
import os
import nltk
import random
import argparse
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
from collections import Counter
from pymorphy2 import MorphAnalyzer
from collections import defaultdict
from timeit import default_timer as timer

nltk.download('punkt_tab')
morph = MorphAnalyzer()


class BM25:
    def __init__(self, k1=1.75, b=0.75):
        """Реализация BM25"""
        self.avgdl = 0
        self.total_docs = 0
        self.k1 = k1
        self.b = b

    def get_dict_freq(self, docs_path: str) -> dict:
        dict_freq = defaultdict(int)
        sum_of_lens = 0
        with open(docs_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc='Считаем частотность слов'):
                if not line.strip():
                    continue
                doc_id, text = line.split('\t')[0], line.split('\t')[3]
                words = text.lower().split()
                sum_of_lens += len(words)
                words = set(words)
                for word in words:
                    dict_freq[word] += 1
                self.total_docs += 1
        self.avgdl = sum_of_lens / self.total_docs
        return dict_freq

    @staticmethod
    def calculate_idf(dict_freq: dict) -> dict:
        dict_idf = {}
        total_docs = len(dict_freq)
        for term, freq in dict_freq.items():
            dict_idf[term] = np.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
        return dict_idf

    def get_score(self, doc: str, query: str, dict_idf: dict) -> float:
        doc_terms = doc.split()
        term_freq = Counter(doc_terms)
        doc_len = len(doc_terms)
        score = 0.0
        for term in query.split():
            if term in term_freq:
                f_t_d = term_freq[term]
                idf_t = dict_idf.get(term, 0)
                score += idf_t * (f_t_d * (self.k1 + 1)) / (
                        f_t_d + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl)))
        return score


def lemmatize_text(text: str) -> str:
    """Лемматризируем текста"""
    tokens = word_tokenize(text)
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return ' '.join(lemmatized_tokens)


def get_dict(docs_path: str, search_column: int, target_column: int, desc: str) -> dict:
    """Функция для того, чтобы считывать определенные столбцы документов в словарь(тут я читаю основной док и запросы)"""
    dict_docs = defaultdict(str)
    with open(docs_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc=desc):
            if not line.strip():
                continue
            id, text = line.split('\t')[search_column].strip(), line.split('\t')[target_column].strip()
            dict_docs[id] = text
    return dict_docs


def get_answer(submission_file: str, dict_queries: dict, dict_docs: dict, dict_idf: dict, bm25) -> dict:
    """Получаем словарь с релевантностью документов"""
    last_query_id = None
    submission_dict = {}
    with open(submission_file, 'r', encoding='utf-8') as sample_submission_file:
        next(sample_submission_file)
        for line in tqdm(sample_submission_file, desc='Получаем скоры для пары запрос-документ'):
            query_id, doc_id = line.split(',')[0].strip(), line.split(',')[1].strip()
            if query_id == last_query_id:
                query = last_query
            else:
                query = lemmatize_text(dict_queries.get(query_id))
                last_query = query
            last_query_id = query_id
            doc = lemmatize_text(dict_docs.get(doc_id))
            score = bm25.get_score(doc, query, dict_idf)
            submission_dict[query_id] = submission_dict.setdefault(query_id, []) + [(doc_id, score)]
        return submission_dict


def create_submision_file(submission_dict, my_submission_file):
    """Сортим словарь с релевантностью документо и создаем submission файл"""
    with open(my_submission_file, 'w', encoding='utf-8') as f:
        f.write('QueryId,DocumentId\n')
        for query_id in tqdm(submission_dict.keys(), desc='Создаем submission файл'):
            sorted_doc_relevance = sorted(submission_dict[query_id], key=lambda x: x[1], reverse=True)
            doc_ids = [pair[0] for pair in sorted_doc_relevance]
            for doc_id in doc_ids:
                f.write(f"{query_id},{doc_id}\n")


def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()
    data_dir = args.data_dir

    # Будем измерять время работы скрипта
    start = timer()
    random.seed(42)

    # определяем нужные пути
    docs_path = os.path.join(data_dir, 'vkmarco-docs.tsv')
    sample_submission_file = os.path.join(data_dir, 'sample_submission.csv')
    queries_path = os.path.join(data_dir, 'vkmarco-doceval-queries.tsv')

    # поехали!
    bm25 = BM25()
    dict_queries = get_dict(queries_path, 0, 1, 'Считываем запросы')
    dict_freq = bm25.get_dict_freq(docs_path)
    dict_idfs = bm25.calculate_idf(dict_freq)
    dict_docs = get_dict(docs_path, 0, 2, 'Считываем документы')
    submission_dict = get_answer(sample_submission_file, dict_queries, dict_docs, dict_idfs, bm25)
    create_submision_file(submission_dict, os.path.join(data_dir, args.submission_file))

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == '__main__':
    main()
