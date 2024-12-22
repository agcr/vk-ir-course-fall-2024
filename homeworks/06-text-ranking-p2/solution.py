#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Set, Tuple

import nltk
import numpy as np
import pandas as pd
import psutil
import pymorphy3 as pymorphy2
from functools import cache
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize()
nltk.download('punkt')
nltk.download('stopwords')
morph = pymorphy2.MorphAnalyzer()


def check_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")
    # if memory_mb > 100:  # 100MB limit
    #     print(f"Memory limit exceeded: {memory_mb:.2f} MB")
    #     sys.exit(1)
    return memory_mb


class TextProcessor:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.russian_stops = set(stopwords.words('russian'))

    def tokenize(self, text: str) -> List[str]:
        return [
            self.normal_form(token) for token in word_tokenize(text.lower(), language='russian')
            if self._is_valid_token(token)
        ]

    def _is_valid_token(self, token: str) -> bool:
        return not (token.isspace() or
                    token in self.russian_stops or
                    set(token).issubset(set('.,!?;:()-')))

    @cache
    def normal_form(self, word: str) -> str:
        return self.morph.parse(word)[0].normal_form


class DataLoader:
    def __init__(self, data_dir: Path, chunksize: int = 1000):
        self.data_dir = data_dir
        self.text_processor = TextProcessor()
        self.chunksize = chunksize

    def process_dataset(self, filename: str):
        output_file = self.data_dir / f"{filename}_tokenized.tsv"

        if output_file.exists():
            return pd.read_csv(output_file, sep='\t', header=None,
                               keep_default_na=False, chunksize=self.chunksize)

        chunks = pd.read_csv(
            self.data_dir / f"{filename}.tsv",
            chunksize=self.chunksize,
            sep='\t',
            header=None,
            keep_default_na=False
        )

        self._process_and_save_chunks(chunks, output_file)
        return pd.read_csv(output_file, sep='\t', header=None,
                           keep_default_na=False, chunksize=self.chunksize)

    def _process_and_save_chunks(self, chunks, output_file: Path):
        for chunk in tqdm(chunks, total=1258656 // self.chunksize, desc='Processing chunks'):
            if 'docs' in output_file.name:
                chunk[2] = chunk[2].parallel_apply(
                    lambda t: ' '.join(self.text_processor.tokenize(t))
                )
                chunk[3] = chunk[3].parallel_apply(
                    lambda t: ' '.join(self.text_processor.tokenize(t))
                )

                chunk[[0, 2, 3]].to_csv(
                    output_file,
                    sep='\t',
                    header=None,
                    index=False,
                    mode='a'
                )
            else:
                chunk[1] = chunk[1].parallel_apply(
                    lambda t: ' '.join(self.text_processor.tokenize(t))
                )
                chunk.to_csv(
                    output_file,
                    sep='\t',
                    header=None,
                    index=False,
                    mode='a'
                )


@dataclass
class StatisticItem:
    doc_length: int
    term_counter: Dict[str, int]
    term_positions: Dict[str, List[int]]

    def __init__(self, text: str):
        text = text.split()
        self.doc_length = len(text)
        self.term_counter = Counter()
        self.term_positions = {}

        for i, term in enumerate(text):
            self.term_counter[term] += 1
            self.term_positions.setdefault(term, []).append(i)

    def term_freq(self, term: str) -> float:
        return self.term_counter.get(term, 0)


@dataclass
class BM25RankerConfig:
    k1: float = 1.2
    b: float = 0.75
    nc: float = 0.8
    title_w: float = 0.9
    body_w: float = 0.7


class BM25Ranker:
    def __init__(self, ranker_config: BM25RankerConfig):
        self.config = ranker_config

    def calculate_scores(self, total_documents, query_docs: Dict, docs_stats: pd.DataFrame,
                         word_freq: Counter, avg_title_length: float, avg_doc_length: float) -> Dict:
        scores = {}
        docs_stats = docs_stats.set_index(0)

        for query_id, (query_text, doc_ids) in tqdm(query_docs.items(),
                                                    desc='Calculating BM25 scores'):
            scores[query_id] = []
            query_terms = query_text.split()

            for doc_id in doc_ids:
                score = self._calculate_doc_score(
                    query_terms=query_terms, title_text=docs_stats.loc[doc_id, 1], doc_text=docs_stats.loc[doc_id, 2],
                    word_freq=word_freq, avg_title_length=avg_title_length,
                    avg_doc_length=avg_doc_length, total_docs=total_documents
                )
                scores[query_id].append((doc_id, score))

        return scores

    @staticmethod
    def _min_distance_between_terms(indexes1, indexes2) -> int:
        min_distance = float('inf')
        i = j = 0

        while i < len(indexes1) and j < len(indexes2):
            distance = abs(indexes1[i] - indexes2[j])
            min_distance = min(min_distance, distance)

            if indexes1[i] < indexes2[j]:
                i += 1
            else:
                j += 1

        return min_distance

    def _calculate_nearness_coefficient(self, doc_stats: StatisticItem, terms: List[str]) -> float:
        if len(terms) <= 1:
            return 1

        pairs = {}
        prev_term = terms[0]
        distances = doc_stats.term_positions

        for term in terms[1:]:
            if term not in distances or prev_term not in distances:
                pairs[(prev_term, term)] = 0
            else:
                pairs[(prev_term, term)] = (doc_stats.doc_length - self._min_distance_between_terms(
                    distances[prev_term], distances[term])) / doc_stats.doc_length
            prev_term = term

        return sum(pairs.values()) / len(pairs)

    def _calculate_doc_score(self, query_terms: List[str],
                             doc_text: str, title_text: str, word_freq: Counter,
                             avg_title_length: float,
                             avg_doc_length: float, total_docs: int) -> float:
        scores = []

        k1 = self.config.k1
        b = self.config.b
        nc = self.config.nc
        title_w = self.config.title_w
        body_w = self.config.body_w

        context = [
            (title_text, avg_title_length),
            (doc_text, avg_doc_length)
        ]

        for text, avg_len in context:
            score = 0
            doc_stats = StatisticItem(text)

            c = self._calculate_nearness_coefficient(doc_stats, query_terms)

            for term in query_terms:
                if term not in word_freq:
                    continue

                f = word_freq[term]
                idf = np.log((total_docs - f + 0.5) / (f + 0.5))
                tf = doc_stats.term_freq(term)
                l = (idf * tf * (k1 + 1) /
                     (tf + k1 * (1 - b + b *
                                 doc_stats.doc_length / avg_len)))

                score += l

            scores.append(score * (nc + c))

        return scores[0] * title_w + scores[1] * body_w


class TextRanking:
    def __init__(self, data_dir: Path, sample_submission: str = 'sample_submission.csv',
                 queries: str = 'vkmarco-doceval-queries', ranker_config: BM25RankerConfig = BM25RankerConfig()):
        self.data_dir = data_dir
        self.data_loader = DataLoader(data_dir, 10000)
        self.bm25_ranker = BM25Ranker(ranker_config)
        self.sample_submission = sample_submission
        self.queries = queries

    def calculate_scores(self, ver='') -> Dict[str, List[Tuple[str, float]]]:
        query_docs, doc_ids = self._load_query_docs()
        docs_texts, word_freq, avg_title_len, avg_doc_len, total_documents = self._load_or_calculate_stats(ver, doc_ids)

        scores = self.bm25_ranker.calculate_scores(
            total_documents,
            query_docs, docs_texts, word_freq,
            avg_doc_length=avg_doc_len, avg_title_length=avg_title_len
        )

        return scores

    def run(self, submission_file: Path):
        scores = self.calculate_scores()
        self._save_submission(scores, submission_file)

    def _load_query_docs(self) -> (Dict, Set):
        submission = self._load_submission()
        doc_ids = {doc_id for docs in submission.values() for doc_id in docs}
        queries = self._load_queries()
        return {
            qid: (queries[qid], docs)
            for qid, docs in submission.items()
        }, doc_ids

    def _load_or_calculate_stats(self, ver, *args, **kwargs) -> Tuple[
        pd.DataFrame, Counter, float, float, int]:
        stats_file = self.data_dir / f'docs_stats_{ver}.csv'
        freq_file = self.data_dir / 'word_freq.pkl'
        avg_len_file = self.data_dir / f'avg_len_{ver}.txt'

        if stats_file.exists() and freq_file.exists() and avg_len_file.exists():
            with open(avg_len_file) as f:
                avg_title_len = float(f.readline())
                avg_doc_len = float(f.readline())
                total_documents = int(f.readline())
            return (
                pd.read_csv(stats_file, header=None, keep_default_na=False),
                pickle.load(open(freq_file, 'rb')),
                avg_title_len,
                avg_doc_len,
                total_documents
            )

        return self._calculate_stats(ver, *args, **kwargs)

    def _calculate_stats(self, ver, doc_ids, save=True, load_freq=True) -> Tuple[
        pd.DataFrame, Counter, float, float, int]:
        if load_freq and (self.data_dir / 'word_freq.pkl').exists():
            word_freq = pickle.load(open(self.data_dir / 'word_freq.pkl', 'rb'))
            update_freq = False
        else:
            word_freq = Counter()
            update_freq = True

        docs_texts = None
        total_title_length = 0
        total_docs_length = 0
        total_docs = 0

        if save and (self.data_dir / f'docs_stats_{ver}.csv').exists():
            os.remove(self.data_dir / f'docs_stats_{ver}.csv')

        for chunk in tqdm(self.data_loader.process_dataset('vkmarco-docs'),
                          total=1258656 // self.data_loader.chunksize,
                          desc='Calculating stats'):

            total_docs += len(chunk)

            if update_freq:
                chunk[1].apply(lambda words: word_freq.update(set(words.split())))
                chunk[2].apply(lambda words: word_freq.update(set(words.split())))

            chunk = chunk[chunk[0].isin(doc_ids)]

            def update_lengths(row):
                nonlocal total_docs_length, total_title_length
                total_title_length += len(row[1].split())
                total_docs_length += len(row[2].split())

            chunk[[1, 2]].apply(lambda row: update_lengths(row), axis=1)

            if save:
                chunk.to_csv(self.data_dir / f'docs_stats_{ver}.csv', header=None, index=False, mode='a')

            if docs_texts is None:
                docs_texts = chunk
            else:
                docs_texts = pd.concat([docs_texts, chunk], ignore_index=True)

        if save:
            with open(self.data_dir / f'avg_len_{ver}.txt', 'w') as f:
                f.writelines([f'{total_title_length / len(docs_texts)}\n',
                              f'{total_docs_length / len(docs_texts)}\n',
                              f'{total_docs}\n'])

            pickle.dump(word_freq, open(self.data_dir / 'word_freq.pkl', 'wb'))

        return docs_texts, word_freq, total_title_length / len(docs_texts), total_docs_length / len(
            docs_texts), total_docs

    def _load_submission(self) -> Dict[str, List[str]]:
        submission = {}
        with open(self.data_dir / self.sample_submission) as f:
            next(f)
            for line in f:
                query_id, doc_id = line.strip().split(',')
                submission.setdefault(int(query_id), []).append(doc_id)
        return submission

    def _load_queries(self) -> Dict[str, str]:
        queries = {}
        for chunk in self.data_loader.process_dataset(self.queries):
            queries.update(dict(zip(chunk[0], chunk[1])))
        return queries

    def _save_submission(self, scores: Dict, submission_file: Path):
        with open(submission_file, 'w') as f:
            f.write('QueryId,DocumentId\n')
            for query_id, docs in sorted(scores.items(),
                                         key=lambda x: int(x[0])):
                for doc_id, _ in sorted(docs,
                                        key=lambda x: x[1],
                                        reverse=True):
                    f.write(f'{query_id},{doc_id}\n')


def main():
    parser = argparse.ArgumentParser(description='Text ranking implementation')
    parser.add_argument('--submission_file', required=True,
                        help='Output Kaggle submission file')
    parser.add_argument('data_dir', help='Input data directory')
    args = parser.parse_args()

    start = timer()
    np.random.seed(42)

    ranker = TextRanking(Path(args.data_dir))

    ranker.run(Path(args.submission_file))

    print(f"Finished in {timer() - start:.3f} seconds")


if __name__ == "__main__":
    main()
