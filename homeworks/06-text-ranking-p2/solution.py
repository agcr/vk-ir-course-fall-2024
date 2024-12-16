#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution"""

import argparse
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from tqdm import tqdm
from pathlib import Path
from numba import njit
from collections import Counter
from nltk.tokenize import WordPunctTokenizer

# Параметры BM25
k1 = 1.5
b = 0.75
tokenizer = WordPunctTokenizer()


@njit
def compute_idf(num_docs, doc_freq):
    """Вычисление IDF для каждого термина."""
    idf = np.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5)) + 1
    return idf


def bm25_score(doc_lengths: list[int], avgdl: float, term_frequencies, idf_values, query_terms: pd.Series):
    """
    Вычисление BM25 для одного документа.
    doc_lengths: длины документов.
    avgdl: средняя длина документа.
    term_frequencies: частота термина в документе.
    idf_values: IDF термина.
    query_terms: список терминов запроса.
    """
    scores = np.zeros(len(doc_lengths))

    for i in range(len(doc_lengths)):
        cur_query = tokenizer.tokenize(str(query_terms.iloc[i]))
        score = 0.0
        doc_len = doc_lengths[i]
        for term in cur_query:
            if term in term_frequencies[i]:
                f = term_frequencies[i][term]
                idf = idf_values.get(term, 0.0)
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / avgdl))
        scores[i] = score
    return scores


# Пример: процесс обработки документов батчами
def process_documents_in_batches(docs, queries, batch_size=1000):
    """Обработка документов по батчам."""
    doc_lengths = []
    term_frequencies = []
    global_frequencies = Counter()
    # Построение частот токенов и вычисление средней длины документа
    for doc in tqdm(docs):
        tokens = tokenizer.tokenize(str(doc).lower())
        doc_lengths.append(len(tokens))
        term_freq = Counter(tokens)
        term_frequencies.append(term_freq)
        global_frequencies.update(term_freq.keys())

    avgdl = np.mean(doc_lengths)
    num_docs = len(doc_lengths)

    # Вычисление IDF для каждого термина
    idf_values = {term: compute_idf(num_docs, global_frequencies[term]) for term in global_frequencies}

    # Итерация по запросам и документам
    results = []
    for start in tqdm(range(0, num_docs, batch_size)):
        end = min(start + batch_size, num_docs)
        batch_scores = bm25_score(
            doc_lengths[start:end],
            avgdl,
            term_frequencies[start:end],
            idf_values,
            queries[start:end]
        )
        results.extend(batch_scores)
    return results


def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
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

    # Дальше вы должны:
    # - загрузить датасет VK MARCO из папки args.data_dir
    # - реализовать какой-то из классических алгоритмов текстового ранжирования, например TF-IDF или BM25
    # - при необходимости, подобрать гиперпараметры с использованием трейна или валидации
    # - загрузить пример сабмишна из args.data_dir/sample_submission.csv
    # - применить алгоритм ко всем запросам и документам из примера сабмишна
    # - переранжировать документы из примера сабмишна в соответствии со скорами, которые выдает ваш алгоритм
    # - сформировать ваш сабмишн для заливки на Kaggle и сохранить его в файле args.submission_file

    data_dir = Path(args.data_dir)

    full_text_chunks = pd.read_csv(
        filepath_or_buffer=data_dir / "vkmarco-docs.tsv",
        sep='\t',
        header=None,
        names=["DocumentId", "URL", "TitleText", "BodyText"],
        chunksize=10000,
        engine='c')
    train_querries = pd.read_csv(
        filepath_or_buffer=data_dir / "vkmarco-doctrain-queries.tsv",
        sep='\t',
        header=None,
        names=["QueryId", "QueryText"],
        chunksize=10000,
        engine='c')

    query_to_doc = pd.read_csv(
        filepath_or_buffer=data_dir / "sample_submission.csv",
        sep=',',
        header="infer",
        engine='c')
    queries = pd.read_csv(
        filepath_or_buffer=data_dir / "vkmarco-doceval-queries.tsv",
        sep='\t',
        header=None,
        names=["QueryId", "QueryText"],
        engine='c')

    query_to_doc = query_to_doc.merge(queries, how="left", on="QueryId")
    unique_texts = set(query_to_doc["DocumentId"].unique())
    needed_texts = []

    for frame in tqdm(full_text_chunks):
        for row in frame.itertuples():
            if row.DocumentId in unique_texts:  # type: ignore
                if row.TitleText:
                    string = ' '.join([str(row.TitleText), str(row.BodyText)])  # type: ignore
                elif row.BodyText:
                    string = str(row.BodyText)
                needed_texts.append({"DocumentId": row.DocumentId, "text": string})  # type: ignore

    final_df = pd.DataFrame(needed_texts).merge(query_to_doc, how="right", on="DocumentId")

    scores = process_documents_in_batches(final_df['text'], final_df['QueryText'])
    final_df["metric"] = scores

    final_df = final_df.sort_values(by=['QueryId', "metric"], ascending=[True, False])
    submit_df = final_df[["QueryId", "DocumentId"]]
    submit_df.reset_index(inplace=True, drop=True)
    submit_df.to_csv(args.submission_file, index=False)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
