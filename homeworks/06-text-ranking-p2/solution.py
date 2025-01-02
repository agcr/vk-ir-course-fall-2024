#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution"""

import argparse
from timeit import default_timer as timer

import random

import math
import pathlib
import shutil
import pandas as pd
from timeit import default_timer as timer
from nltk import tokenize
import pymorphy3 as pymorphy

# global tokenizer and lemmatizer
morph = pymorphy.MorphAnalyzer(lang='ru')
tokenizer = tokenize.RegexpTokenizer(r'\w+')


class ManualTfidfVectorizer:
    """
    Uses IDF definition: IDF(t) = log ( (N - DF(t) + 0.5) / (DF(t) + 0.5) )
    """
    def __init__(self):
        self.df = {}   # term: int
        self.nbdocs = 0
        self.counts = {'title': 0, 'body': 0, 'total': 0}

    def fit(self, texts):
        for doc in texts:
            for term in set(doc['title'] + doc['body']):
                term_counter = self.df.setdefault(term, 0)
                self.df[term] = term_counter + 1
            count_title, count_body = len(doc['title']), len(doc['body'])
            self.counts['title'] += count_title
            self.counts['body'] += count_body
            self.counts['total'] += count_title + count_body
        self.nbdocs += len(texts)

    def get_idf(self, word):
        df = self.df.get(word, 0)
        return math.log((self.nbdocs - df + 0.5) / (df + 0.5))

    def get_avg(self):
        return {k: i/self.nbdocs for k, i in self.counts.items()}


def preprocess(text):
    tokens = tokenizer.tokenize(text)
    return [morph.parse(token.lower())[0].normal_form for token in tokens]


def read_docs(docs_file, sorted_docs_uniq):
    """
    Функция, которая читает все документы в один большой список
    примерно 16 часов продолжается
    """
    docs = []
    j, num_line = 0, 0   # j - start point
    nb_sorted_docs = len(sorted_docs_uniq)
    start = timer()
    progress_every = nb_sorted_docs // 100
    with open(docs_file, 'rt', encoding='utf-8') as f:
        for line in f:
            # Парсим следующую строку
            parts = line.rstrip('\n').split('\t')
            docid, url, title, body = parts

            # Валидируем
            if not docid:
                raise RuntimeError(f"invalid doc id: num_lines = {num_line}")
            if not url:
                raise RuntimeError(f"invalid url: num_lines = {num_line}")
            num_line = int(docid[1:]) + 1
            
            # фильтруем
            if docid > sorted_docs_uniq[j]:
                while j < nb_sorted_docs and docid > sorted_docs_uniq[j]:
                    j += 1
                if j == nb_sorted_docs:
                    break
            if docid < sorted_docs_uniq[j]:
                continue

            # Пакуем данные документа в словарь
            doc = {'url': url, 'title': preprocess(title), 'body': preprocess(body), 'docid': docid}
            docs.append(doc)

            # display progress
            if (j+1) % progress_every == 0 or (j+1) == nb_sorted_docs:
                elapsed = timer() - start
                remaining = elapsed / (j + 1) * nb_sorted_docs - elapsed
                print(f'\rProcessed {j+1:6d} of {nb_sorted_docs} docs ({(j+1)/nb_sorted_docs:.0%}). '
                      f'Remaining est. {remaining/60:.0f} minute(s)    ', end='', flush=True)
    
    h, m, s = elapsed // 3600, elapsed % 3600 // 60, elapsed % 60
    print(f'\nFinished in {h:.0f} hours {m:.0f} minutes {s:.0f} seconds')        
    return docs


def bm25f(query, document, vectorizer, lavg, wzones={'title':8, 'body':1}, k=1.2, b=0.75, bz=0.75):
    terms = preprocess(query)
    bm25f_score = 0
    l = 0
    lz = {}
    for key in wzones:
        size = len(document[key])
        lz[key] = size
        l += size
    
    for term in terms:
        tw = 0
        l = 0
        for zone, weight in wzones.items():
            tf = document[zone].count(term)
            tw += weight * tf / (1 - bz + bz * (lz[zone] / lavg[zone]))
        bm25f_score += vectorizer.get_idf(term) * tw * (k + 1) / (tw + k * (1 - b + b * l / lavg['total']))
    return bm25f_score


def find_document(doc_id, doc_list, nb_docs, full_doc_list):
    
    l, r = -1, nb_docs
    while r - l > 1:
        m = (l + r) // 2
        current_docid = doc_list[m]
        if current_docid == doc_id:
            return full_doc_list[m]
        elif current_docid < doc_id:
            l = m
        else:
            r = m
    print("Document", doc_id, "not found.")    # так не должно быть, но поставил проверку на всякий случай
    return {}


def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Будем измерять время работы скрипта
    start = timer()

    # Тут вы, скорее всего, должны проинициализировать фиксированным seed'ом генератор случайных чисел для того,
    #  чтобы ваше решение было воспроизводимо.
    # Например (зависит от того, какие конкретно либы вы используете):
    # random.seed(2314)
    # np.random.seed(42)
    # и т.п.

    # Дальше вы должны:
    # - загрузить датасет VK MARCO из папки args.data_dir
    data_dir = pathlib.Path(args.data_dir)
    
    # Файл с запросами
    queries_file = data_dir.joinpath("vkmarco-doceval-queries.tsv")
    # Загружаем запросы во фрейм
    queries_df = pd.read_csv(queries_file, sep='\t', header=None)
    queries_df.columns = ['query_id', 'query_text']

    # Представляем запросы в виде словаря: Query ID -> Text
    query_id_to_text = {query_id: query_text for query_id, query_text in zip(queries_df['query_id'], queries_df['query_text'])}

    # Файл без оценок
    samplesub_file = data_dir.joinpath("sample_submission.csv")

    # Загружаем пару запроса документов во фрейм
    samplesub_df = pd.read_csv(samplesub_file)

    # Получаем pandas series уникальних документов из sample_submission
    sorted_docs_uniq = sorted(samplesub_df['DocumentId'].unique().tolist())
    nb_docs_uniq = len(sorted_docs_uniq)

    # Файл с документами
    docs_file = data_dir.joinpath("vkmarco-docs.tsv")

    # Загружаем те документы, которые лежат в файле sample_submission.csv
    docs = read_docs(docs_file, sorted_docs_uniq)
    print(f"Loaded {len(docs)} docs")

    # - реализовать какой-то из классических алгоритмов текстового ранжирования, например TF-IDF или BM25
    vectorizer = ManualTfidfVectorizer()
    # Примерно две минуты для векторизации
    startv = timer()
    print("Vectorizing documents")
    vectorizer.fit(docs)
    elapsedv = timer() - startv
    print(f"Finished in {elapsedv:.3f} secs")
    lavg = vectorizer.get_avg()

    # - при необходимости, подобрать гиперпараметры с использованием трейна или валидации
    # - загрузить пример сабмишна из args.data_dir/sample_submission.csv
    # - применить алгоритм ко всем запросам и документам из примера сабмишна
    # примерно 8 минут 
    start_resp = timer()
    print("Applying scores")
    samplesub_df['score'] = samplesub_df.apply(lambda x: bm25f(
        query_id_to_text[x['QueryId']],
        find_document(x['DocumentId'], sorted_docs_uniq, nb_docs_uniq, docs),
        vectorizer,
        lavg), axis=1)
    print(f'All done in {timer() - start_resp:.3f} secs')
    # print(samplesub_df.head(10))

    # - переранжировать документы из примера сабмишна в соответствии со скорами, которые выдает ваш алгоритм
    result_df = samplesub_df.sort_values(by=['QueryId', 'score'], ascending=[True, False])

    # - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file
    result_df[['QueryId', 'DocumentId']].to_csv(args.submission_file, index=False)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
