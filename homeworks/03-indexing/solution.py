#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import tracemalloc
from tqdm import tqdm
from nltk import tokenize
from functools import reduce
from timeit import default_timer as timer


def measure_memory_usage(func):
    """Measure memory usage of `func`"""

    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        peak_memory = tracemalloc.get_traced_memory()[1]
        peak_memory_mb = peak_memory / 1024 / 1024
        print(f"Максимальное потребление памяти в {func.__name__}: {peak_memory_mb:.2f} МБ")
        tracemalloc.stop()
        return result

    return wrapper


def load_document(input_file):
    """Load document from file"""
    with open(input_file, 'r') as file:
        for line in file:
            yield line.strip()


def preprocess(text):
    """Preprocess text: Tokenize and Normalize"""
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return [token.lower() for token in tokens]


def save_to_output(term, positions_list, file):
    """Функция для сохранения данных в файл обратного индекса"""
    file.write(f"{term}\t")
    file.write(','.join(map(str, sorted(positions_list))))
    file.write("\n")


def generate_term_docid_pairs(documents, output_file):
    """Маппим документ → список пар (термин, DocID) и сразу запись на диск, чтобы не накапливать данные в памяти"""
    with open(output_file, 'w') as out_file:
        for _, line in tqdm(enumerate(documents), desc="Создаем индекс"):
            document = line.strip().split('\t')
            doc_id, doc_title, doc_body = document[0], document[2], document[3]
            terms = preprocess(doc_title + ' ' + doc_body)
            for term in terms:
                out_file.write(f"{term}\t{doc_id}\n")


def sort_term_docid_pairs(input_file, output_file):
    """Сортируем пары на диске, в качестве ключа используем термин"""
    command = f"sort -k1,1 {input_file} > {output_file}"
    os.system(command)


def group_and_save_index(input_file_path, output_file_path):
    """Сохраняем индекс на диск"""
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        current_term = None
        positions_list = []
        for line in input_file:
            term, doc_id = line.strip().split('\t')
            if current_term is not None and term != current_term:
                save_to_output(current_term, positions_list, output_file)
                positions_list.clear()
            positions_list.append(doc_id)
            current_term = term

        if current_term is not None:
            save_to_output(current_term, positions_list, output_file)


def get_document_ids_for_terms(file_path, terms):
    """Функция, которая возвращает словарь из терминов и списков документов к нему"""
    term_to_docs = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                term = parts[0]
                doc_ids_str = parts[1].strip()
                if not doc_ids_str:
                    continue
                doc_ids = set(doc_ids_str.split(','))
                if term in terms:
                    term_to_docs[term] = doc_ids

    return term_to_docs


def search_by_index(term_to_docs):
    """Находим пересечение всех наборов документов"""
    all_doc_lists = term_to_docs.values()
    intersection = reduce(set.intersection, map(set, all_doc_lists)) if all_doc_lists else set()
    return intersection


def get_answer(file_query, file_objects, submission_file, index_file):
    last_query_id = None
    with open(file_query, 'r', encoding='utf-8') as file_query:
        with open(file_objects, 'r', encoding='utf-8') as file_objects:
            _ = file_objects.readline()
            with open(submission_file, 'w', encoding='utf-8') as file_submission:
                file_submission.write('ObjectId,Label\n')
                for line in tqdm(file_objects, desc="Ищем релевантные документы"):
                    objct = line.strip().split(',')
                    obj_id, query_id, doc_id = objct
                    if query_id != last_query_id:
                        query_text = file_query.readline().strip().split('\t')[1]
                        query_tokens = preprocess(query_text)
                        dict_ids = get_document_ids_for_terms(index_file, query_tokens)
                        docs_id = search_by_index(dict_ids)
                        last_query_id = query_id

                    if doc_id in docs_id:
                        file_submission.write(f"{obj_id},1\n")
                    else:
                        file_submission.write(f"{obj_id},0\n")


@measure_memory_usage
def main():
    """
    В общем, мое решение может выглядеть костыльно, а может так оно и есть)), но я делала чисто по лекции
    Состоит из 3-х этапов:
     MAP: документ → список пар (термин, DocID)
     SORT: сортируем пары на диске, в качестве ключа используем термин (причем тут вы сказали про то, что можно
    на bash сортить пары на диске)
     REDUCE: группируем DocID’ы одного и того же термина в списки словопозиций
    """
    parser = argparse.ArgumentParser(description='Indexing homework solution')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('--build_index', action='store_true', help='force reindexing')
    parser.add_argument('--index_dir', required=True, help='index directory')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()
    start = timer()
    if args.build_index:
        if not os.path.exists(args.index_dir):
            os.makedirs(args.index_dir)

        pairs_file = os.path.join(args.index_dir, 'term_docid_pairs.txt')
        sort_pairs_file = os.path.join(args.index_dir, 'sort_pairs.txt')

        documents = load_document(os.path.join(args.data_dir, 'vkmarco-docs.tsv'))
        generate_term_docid_pairs(documents, pairs_file)
        sort_term_docid_pairs(pairs_file, sort_pairs_file)
        group_and_save_index(sort_pairs_file, os.path.join(args.index_dir, 'invert_index.txt'))
    else:
        get_answer(os.path.join(args.data_dir, 'vkmarco-doceval-queries.tsv'),
                   os.path.join(args.data_dir, 'objects.csv'),
                   args.submission_file,
                   os.path.join(args.index_dir, 'invert_index.txt'))

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
