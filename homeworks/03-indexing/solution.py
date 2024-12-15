#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
from timeit import default_timer as timer
import pickle
from nltk import tokenize


def build_index(docs):
    inverted_index = {}
    next_doc_id = 1
    for doc in docs:
        for word in doc:
            postings = inverted_index.setdefault(word, set())
            postings.add(next_doc_id)
        next_doc_id += 1
    return inverted_index


def find_priorities(text_list):
    lvl, total_open, total_closed = 0, 0, 0
    leveled_list = []
    for element in text_list:
        if lvl > (len(leveled_list) - 1):
            leveled_list.append([])
        if element == '(':
            total_open += 1
            leveled_list[lvl].append(total_open)
            lvl = total_open
            continue
        elif element == ')':
            total_closed += 1
            lvl = total_open - total_closed
            continue
        leveled_list[lvl].append(element)
        
    return leveled_list


class Node():
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right


def build_tree(priority_list, base_list=None):

    if not base_list:
        base_list = priority_list[0]
    base_list_size = len(base_list)
    if base_list_size > 1:
        skipop = 0
        divindex = base_list_size - 1
        # let's find the first 'or' operator
        oridx = base_list_size - 2
        while oridx > 0:
            
            if base_list[oridx] == '|':
                skipop = 1
                operator = '|'
                divindex = oridx
                break
            oridx -= 1
        else:
            operator = '&'

        if len(base_list[:divindex]) == 1:
            if isinstance(base_list[0], int):
                left = build_tree(priority_list, priority_list[base_list[0]])
            else:
                left = base_list[0]
        else:
            left = build_tree(priority_list, base_list[:divindex])
        if len(base_list[divindex+skipop:]) == 1:
            if isinstance(base_list[-1], int):
                right = build_tree(priority_list, priority_list[base_list[-1]])
            else:
                right = base_list[-1]
        else:
            right = build_tree(priority_list, base_list[divindex+skipop:])
        return Node(operator, left, right)


def base_search(term1, term2, operator='&'):
    return term1 | term2 if operator == '|' else term1 & term2


def search(tree, inverted_index):
    left, right = tree.left, tree.right
    if isinstance(left, Node):
        left = search(left, inverted_index)
    if isinstance(right, Node):
        right = search(right, inverted_index)
    if not isinstance(left, set):
        left = inverted_index.get(left, set())
    if not isinstance(right, set):
        right = inverted_index.get(right, set())
    return base_search(left, right, tree.operator)


def preprocess(text, regexp=r'\w+'):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(regexp)
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]


def display_progress(start, idx, doctype='documents'):
    partial_time = timer() - start
    print(f"\rProcessed {idx:5d} {doctype} in {partial_time:.3f}   ", end='', flush=True)


def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Indexing homework solution')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('--build_index', action='store_true', help='force reindexing')
    parser.add_argument('--index_dir', required=True, help='index directory')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Будем измерять время работы скрипта
    start = timer()

    # Какой у нас режим: построения индекса или генерации сабмишна?
    if args.build_index:
        # Тут вы должны:
        # - загрузить тексты документов из файла args.data_dir/vkmarco-docs.tsv
        print("Opening docs file")
        docs_file = f'{args.data_dir}/docs.txt'
        # - проиндексировать эти тексты, причем для разбиения текстов на слова (термины) надо воспользоваться функцией preprocess()
        token_docs = []
        with open(docs_file, encoding='utf-8') as docs:
            for line in docs.readlines():
                # we consider title and body
                docid, title, body = line.rstrip('\n').split('\t')
                token_docs.append(preprocess(title + ' ' + body))
                # token_docs.append(preprocess(body))
                
                docidx = int(docid[1:])
                if docidx % 100 == 0:
                    display_progress(start, docidx, 'documents')
        display_progress(start, docidx, 'documents')
        print("\nSaving inverted index doc file")
        # - сохранить получивший обратный индекс в папку переданную через параметр args.index_dir
        docs_index = f'{args.index_dir}/docs_index.pkl'
        with open(docs_index, 'wb') as file:
            # file.write(str(build_index(token_docs)))
            pickle.dump(build_index(token_docs), file)
        
        # pass
    else:
        print("Opening inverted index doc file")
        docs_index = f'{args.index_dir}/docs_index.pkl'
        with open(docs_index, 'rb') as file:
            # inverted_index = [f.rstrip() for f in file.readlines()]
            inverted_index = pickle.load(file)

        # Тут вы должны:
        # - загрузить поисковые запросы из файла args.data_dir/vkmarco-doceval-queries.tsv
        # - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
        # - для разбиения текстов запросов на слова тоже используем функцию preprocess()
        queries_file = f'{args.data_dir}/queries.numerate.txt'
        result = []

        with open(queries_file, encoding='utf-8') as queries:
            for query in queries.readlines():                
                queryidx, query_text = query.rstrip().split('\t')
                preprocessed_query = preprocess(query_text, r'\w+|[\(\)\|]')
                query_tree = build_tree(find_priorities(preprocessed_query))

                if query_tree:
                    result.append(search(query_tree, inverted_index))
                else:
                    result.append(inverted_index.get(query_text, set()))

                queryidx = int(queryidx)
                if queryidx % 100 == 0:
                    display_progress(start, queryidx, 'queries')
        
        # print('\nfirst part end')
        # - сформировать ваш сабмишн, в котором для каждого объекта (пары запрос-документ) 
        # будет проставлена метка 1 (в документе есть все слова из запроса) или 0
        #
        # Для формирования сабмишна надо загрузить и использовать файлы args.data_dir/sample_submission.csv и args.data_dir/objects.csv
        # , encoding='utf-8'
        print("\nGenerating submission file")
        obj_numerate = f'{args.data_dir}/objects.numerate.txt'
        submission_file = f'{args.data_dir}/submission.csv'
        total_res = 0
        with open(obj_numerate, encoding='utf-8') as objects, open(submission_file, 'w', encoding='utf-8') as fsub:
            next(objects)   # skip header
            fsub.write('ObjectId,Relevance\n')
            for obj in objects.readlines():
                obj_id, query_id, document_id = obj.rstrip().split(',')
                res = 1 if int(document_id[1:]) in result[int(query_id)-1] else 0
                total_res += res
                fsub.write(f'{obj_id},{res}\n')
        print(f"Total of {total_res} pairs queries/documents found.")
        # pass

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
