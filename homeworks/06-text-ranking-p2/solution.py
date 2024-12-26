#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution"""

import argparse
import os
import csv
import random
import numpy as np
from timeit import default_timer as timer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Будем измерять время работы скрипта
    start = timer()

    random.seed(42)
    np.random.seed(42)

    docs_path = os.path.join(args.data_dir, 'vkmarco-docs.tsv')
    doc_texts = {}

    csv.field_size_limit(2 ** 31 - 1)

    with open(docs_path, 'r', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for row in tsv_reader:
            doc_id = row[0]
            title = row[2]
            body = row[3]
            text = f"{title} {body}"
            doc_texts[doc_id] = text

    queries_path = os.path.join(args.data_dir, 'vkmarco-doceval-queries.tsv')
    query_texts = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for row in tsv_reader:
            query_id = row[0]
            query_text = row[1]
            query_texts[query_id] = query_text

    sample_sub_path = os.path.join(args.data_dir, 'sample_submission.csv')
    queries_to_docs = {}
    with open(sample_sub_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            q_id = row[0]
            d_id = row[1]
            if q_id not in queries_to_docs:
                queries_to_docs[q_id] = []
            queries_to_docs[q_id].append(d_id)

    ranking_results = {}

    vectorizer = TfidfVectorizer(

    )

    for q_id, doc_ids in queries_to_docs.items():
        combined_texts = []
        combined_texts.append(query_texts[q_id])

        for d_id in doc_ids:
            if d_id in doc_texts:
                combined_texts.append(doc_texts[d_id])
            else:
                combined_texts.append("")

        tfidf_matrix = vectorizer.fit_transform(combined_texts)

        query_vector = tfidf_matrix[0:1]
        doc_vectors = tfidf_matrix[1:]

        cos_sim = cosine_similarity(query_vector, doc_vectors)[0]

        doc_scores = []
        for i, d_id in enumerate(doc_ids):
            score = cos_sim[i]
            doc_scores.append((d_id, score))

        doc_scores.sort(key=lambda x: x[1], reverse=True)

        ranking_results[q_id] = doc_scores

    with open(args.submission_file, 'w', encoding='utf-8', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['QueryId', 'DocumentId'])

        for q_id in ranking_results:
            sorted_docs = ranking_results[q_id]
            for d_id, _score in sorted_docs:
                writer.writerow([q_id, d_id])

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
