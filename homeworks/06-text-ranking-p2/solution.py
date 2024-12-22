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
from nltk.util import bigrams

CHUNKSIZE = 10000
K1 = 1.2
B = 0.75

# Инициализация токенизатора и лемматизатора
morph = MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))


def preprocess(text):
    """Tokenizes, normalizes the input text, and removes stop words."""
    tokens = tokenizer.tokenize(text.lower())
    normalized_tokens = [morph.normal_forms(token)[0] for token in tokens]
    
    # Убрать стоп-слова
    filtered_tokens = [token for token in normalized_tokens if token not in stop_words]
    
    # Генерация биграм
    token_bigrams = ['_'.join(bigram) for bigram in bigrams(filtered_tokens)]
    
    # Униграммы + биграммы
    return filtered_tokens + token_bigrams

    
def bm25_score(query_counter, doc_counter, doc_length, term_doc_count, total_docs, avg_doc_len, k1=K1, b=B):
    score = 0.0
    b_factor = 1 - b + b * (doc_length / avg_doc_len)

    for term, q_count in query_counter.items():
        tf = doc_counter.get(term, 0)
        doc_freq = term_doc_count.get(term, 0)

        # IDF с проверкой doc_freq
        idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        # Нормализованный tf
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * b_factor) if tf > 0 else 0

        # Добавление вклада терма в общий результат
        score += idf * tf_norm * q_count

    return score


def process_queries(data_dir):
    """Основная функция обработки документов и запросов."""
    docs_file = f"{data_dir}/vkmarco-docs.tsv"
    queries_file = f"{data_dir}/vkmarco-doceval-queries.tsv"
    submission_sample = f"{data_dir}/sample_submission.csv"

    submission = pd.read_csv(submission_sample)
    relevant_doc_ids = set(submission['DocumentId'].astype(str))

    # Обработка документов по чанкам
    document_chunks = pd.read_csv(
        docs_file, sep='\t', names=['DocumentId', 'URL', 'TitleText', 'BodyText'], chunksize=CHUNKSIZE
    )

    term_frequencies = {}
    term_document_frequency = defaultdict(int)
    document_count = 0
    total_document_length = 0

    for chunk_index, chunk in enumerate(document_chunks, start=1):
        print(f"Processing chunk {chunk_index}...")
        chunk['DocumentId'] = chunk['DocumentId'].astype(str)
        chunk_filtered = chunk[chunk['DocumentId'].isin(relevant_doc_ids)]

        for _, row in chunk_filtered.iterrows():
            doc_id = row['DocumentId']
            text = f"{row['TitleText']} {row['BodyText']}"
            tokens = preprocess(text)

            doc_length = max(len(tokens), 1)
            doc_counter = Counter(tokens)

            term_frequencies[doc_id] = (doc_counter, doc_length)
            for term in set(tokens):
                term_document_frequency[term] += 1

            document_count += 1
            total_document_length += doc_length

    avg_doc_len = total_document_length / document_count if document_count > 0 else 1

    print("All chunks processed! Starting queries processing.")

    # Обработка запросов
    queries = pd.read_csv(queries_file, sep='\t', names=['QueryId', 'QueryText'])
    results = []

    for query_index, (query_id, query_text) in enumerate(zip(queries['QueryId'], queries['QueryText']), start=1):
        query_tokens = preprocess(query_text)
        query_counter = Counter(query_tokens)

        candidates = submission[submission['QueryId'] == query_id]
        scores = {}

        for _, row in candidates.iterrows():
            doc_id = row['DocumentId']
            doc_info = term_frequencies.get(doc_id, (Counter(), 1))
            doc_counter, doc_length = doc_info

            # Рассчёт BM25
            score_value = bm25_score(
                query_counter, doc_counter, doc_length, term_document_frequency, document_count, avg_doc_len
            )
            scores[doc_id] = score_value

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results.extend([(query_id, doc_id) for doc_id, _ in sorted_docs])

        if query_index % 100 == 0:
            print(f"Processed {query_index} queries!")

    return results

def main():
    parser = argparse.ArgumentParser(description='Text ranking homework solution with BM25')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()
    results = process_queries(data_dir=args.data_dir)
    pd.DataFrame(results, columns=['QueryId', 'DocumentId']).to_csv(args.submission_file, index=False)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")

if __name__ == "__main__":
    main()