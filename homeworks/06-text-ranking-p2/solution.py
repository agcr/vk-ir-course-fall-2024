#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution with BM25"""

import argparse
from timeit import default_timer as timer
from nltk.tokenize import RegexpTokenizer
from pymorphy3 import MorphAnalyzer
from collections import Counter, defaultdict
import pandas as pd
import math
from joblib import Parallel, delayed
import nltk
from nltk.corpus import stopwords

# Инициализация токенизатора и лемматизатора
morph = MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

def preprocess(text):
    """Tokenizes, normalizes the input text, and removes stop words."""
    tokens = tokenizer.tokenize(text.lower())
    normalized_tokens = [morph.normal_forms(token)[0] for token in tokens]
    return [token for token in normalized_tokens if token not in stop_words]

def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)

def process_chunk(chunk, relevant_doc_ids, chunk_index, total_chunks):
    print(f"Processing chunk {chunk_index}/{total_chunks}...")
    chunk['DocumentId'] = chunk['DocumentId'].astype(str)
    chunk_filtered = chunk[chunk['DocumentId'].isin(relevant_doc_ids)]

    local_tf = {}
    local_term_doc_count = defaultdict(int)
    local_total_docs = 0

    for _, row in chunk_filtered.iterrows():
        doc_id = row['DocumentId']
        text = f"{row['TitleText']} {row['BodyText']}"
        tokens = preprocess(text)

        doc_length = len(tokens)
        if doc_length == 0:
            doc_length = 1

        doc_counter = Counter(tokens)
        local_tf[doc_id] = (doc_counter, doc_length)

        for term in set(tokens):
            local_term_doc_count[term] += 1

        local_total_docs += 1

    return local_tf, local_term_doc_count, local_total_docs

def calculate_bm25_score(query_counter, doc_counter, doc_length, term_doc_count, total_docs, avg_doc_len, k1=1.2, b=0.75):
    """
    Рассчитывает BM25 для пары (запрос, документ).
    query_counter: Counter термов запроса
    doc_counter: Counter термов документа
    doc_length: длина документа
    term_doc_count: словарь term->doc_freq
    total_docs: общее число документов
    avg_doc_len: средняя длина документа
    """
    score = 0.0
    for term, q_count in query_counter.items():
        tf = doc_counter.get(term, 0)
        doc_freq = term_doc_count.get(term, 0)

        # IDF
        idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        # Нормализованный tf с учётом длины документа
        denom = tf + k1*(1 - b + b*(doc_length/avg_doc_len))
        tf_norm = ((tf*(k1+1)) / denom) if denom != 0 else 0

        # Добавляем в скор (учитываем q_count, но обычно q_count в запросе = 1 для большинства случаев)
        score += idf * tf_norm * q_count

    return score

def process_docs_and_queries(data_dir, submission_file):
    docs_file = f"{data_dir}/vkmarco-docs.tsv"
    queries_file = f"{data_dir}/vkmarco-doceval-queries.tsv"
    submission_sample = f"{data_dir}/sample_submission.csv"

    submission = pd.read_csv(submission_sample)
    relevant_doc_ids = set(submission['DocumentId'].astype(str))

    total_lines = count_lines(docs_file)
    chunksize = 1000
    total_chunks = (total_lines + chunksize - 1) // chunksize

    doc_reader = pd.read_csv(
        docs_file, sep='\t', names=['DocumentId', 'URL', 'TitleText', 'BodyText'], chunksize=chunksize
    )

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_chunk)(chunk, relevant_doc_ids, i, total_chunks) 
        for i, chunk in enumerate(doc_reader, start=1)
    )

    tf = {}
    term_doc_count = defaultdict(int)
    total_docs = 0
    sum_doc_length = 0  # для вычисления средней длины документа

    for local_tf, local_term_doc_count, local_total_docs in results:
        tf.update(local_tf)
        for term, count in local_term_doc_count.items():
            term_doc_count[term] += count
        total_docs += local_total_docs

    # Вычисляем среднюю длину документа
    for doc_id, (doc_counter, doc_length) in tf.items():
        sum_doc_length += doc_length
    avg_doc_len = sum_doc_length / total_docs if total_docs > 0 else 1

    print("All chunks processed! Starting queries processing.")

    queries = pd.read_csv(queries_file, sep='\t', names=['QueryId', 'QueryText'])
    results = []

    for i, (query_id, query_text) in enumerate(queries.itertuples(index=False)):
        query_tokens = preprocess(query_text)
        query_counter = Counter(query_tokens)

        candidates = submission[submission['QueryId'] == query_id]
        scores = {}

        for _, row in candidates.iterrows():
            doc_id = row['DocumentId']
            doc_info = tf.get(doc_id, (Counter(), 1)) 
            doc_counter, doc_length = doc_info
            # Считаем BM25 скор
            bm25_score = calculate_bm25_score(query_counter, doc_counter, doc_length, term_doc_count, total_docs, avg_doc_len)
            scores[doc_id] = bm25_score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results.extend([(query_id, doc_id) for doc_id, _ in sorted_docs])

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} queries!")

    pd.DataFrame(results, columns=['QueryId', 'DocumentId']).to_csv(submission_file, index=False)
    print("Results saved!")

def main():
    parser = argparse.ArgumentParser(description='Text ranking homework solution with BM25')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()
    process_docs_and_queries(args.data_dir, args.submission_file)
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")

if __name__ == "__main__":
    main()
