#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution"""

import argparse
from timeit import default_timer as timer
from typing import Iterable

import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import WordPunctTokenizer
from tqdm.notebook import tqdm
from numba import njit

# Constants
CHUNK_SIZE = 10000
K1 = 2.0
B = 0.75

# Tokenizer
TOKENIZER = WordPunctTokenizer()


@njit
def calculate_idf(total_documents: int, document_frequency: int) -> float:
    return np.log((total_documents - document_frequency + 0.5) / (document_frequency + 0.5)) + 1


def load_datasets(data_dir: str) -> tuple[Iterable[pd.DataFrame], pd.DataFrame]:
    document_chunks = pd.read_csv(
        f"{data_dir}/vkmarco-docs.tsv",
        sep='\t',
        header=None,
        names=["DocumentId", "URL", "TitleText", "BodyText"],
        chunksize=CHUNK_SIZE,
        engine='c'
    )

    submission_data = pd.read_csv(
        f"{data_dir}/sample_submission.csv",
        sep=',',
        header="infer",
        engine='c'
    )

    queries_data = pd.read_csv(
        f"{data_dir}/vkmarco-doceval-queries.tsv",
        sep='\t',
        header=None,
        names=["QueryId", "QueryText"],
        engine='c'
    )

    submission_data = submission_data.merge(queries_data, how="left", on="QueryId")
    return document_chunks, submission_data


def extract_relevant_documents(document_chunks: Iterable[pd.DataFrame], relevant_document_ids: set) -> list[dict]:
    relevant_documents = []
    for chunk in tqdm(document_chunks):
        for row in chunk.itertuples():
            if row.DocumentId in relevant_document_ids:
                document_text = ' '.join([str(row.TitleText), str(row.BodyText)]) if row.TitleText else str(
                    row.BodyText)
                relevant_documents.append({"DocumentId": row.DocumentId, "text": document_text})
    return relevant_documents


def compute_bm25_scores(document_lengths: list[int], average_document_length: float, term_frequencies: list[Counter],
                        idf_values: dict, query_texts: pd.Series) -> np.ndarray:
    scores = np.zeros(len(document_lengths))
    for i in range(len(document_lengths)):
        query_terms = TOKENIZER.tokenize(str(query_texts.iloc[i]))
        document_length = document_lengths[i]
        score = 0.0
        for term in query_terms:
            if term in term_frequencies[i]:
                term_frequency = term_frequencies[i][term]
                idf = idf_values.get(term, 0.0)
                score += idf * (term_frequency * (K1 + 1)) / (
                            term_frequency + K1 * (1 - B + B * document_length / average_document_length))
        scores[i] = score
    return scores


def process_documents_and_queries(documents: pd.Series, queries: pd.Series, batch_size: int = 1000) -> list[float]:
    document_lengths = []
    document_term_frequencies = []
    global_term_frequencies = Counter()

    for document in tqdm(documents):
        tokens = TOKENIZER.tokenize(str(document).lower())
        document_lengths.append(len(tokens))
        term_frequency = Counter(tokens)
        document_term_frequencies.append(term_frequency)
        global_term_frequencies.update(term_frequency.keys())

    average_document_length = np.mean(document_lengths)
    total_documents = len(document_lengths)

    idf_values = {term: calculate_idf(total_documents, global_term_frequencies[term]) for term in
                  global_term_frequencies}

    scores = []
    for start in tqdm(range(0, total_documents, batch_size)):
        end = min(start + batch_size, total_documents)
        batch_scores = compute_bm25_scores(
            document_lengths[start:end],
            average_document_length,
            document_term_frequencies[start:end],
            idf_values,
            queries[start:end]
        )
        scores.extend(batch_scores)

    return scores


def save_submission_file(results_df: pd.DataFrame, output_file: str) -> None:
    submission_df = results_df[["QueryId", "DocumentId"]]
    submission_df.reset_index(inplace=True, drop=True)
    submission_df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start_time = timer()

    document_chunks, submission_data = load_datasets(args.data_dir)

    relevant_document_ids = set(submission_data["DocumentId"].unique())
    relevant_documents = extract_relevant_documents(document_chunks, relevant_document_ids)

    results_df = pd.DataFrame(relevant_documents).merge(submission_data, how="right", on="DocumentId")

    bm25_scores = process_documents_and_queries(results_df['text'], results_df['QueryText'])
    results_df["BM25Score"] = bm25_scores

    results_df = results_df.sort_values(by=['QueryId', "BM25Score"], ascending=[True, False])
    save_submission_file(results_df, args.submission_file)

    elapsed_time = timer() - start_time
    print(f"Finished processing in {elapsed_time:.3f} seconds")


if __name__ == "__main__":
    main()
