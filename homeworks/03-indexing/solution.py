import argparse
import os
import platform
import shutil
import sys
import gzip
import zipfile
from timeit import default_timer as timer
from pathlib import Path
from timeit import default_timer as timer
from collections import defaultdict


def preprocess(text):
    from nltk import tokenize
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]


def calculate_object_size(obj, visited=None):
    if visited is None:
        visited = set()
    if id(obj) in visited:
        return 0
    visited.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, (dict, list, tuple, set)):
        size += sum(calculate_object_size(item, visited) for item in obj)
    return size


def save_split(split, folder, start, end):
    filename = f"{start}-{end}.tsv"
    with open(os.path.join(folder, filename), 'w') as f:
        for term, docs in split.items():
            f.write(f"{term}\t{' '.join(docs)}\n")


def split_index(index, index_dir, max_size=10 * 1024 * 1024):
    sizes = {
        term: calculate_object_size(doc_ids)
        for term, doc_ids in index.items()
    }
    sizes = dict(sorted(sizes.items(), key=lambda item: item[0], reverse=True))
    start_term = None
    end_term = None
    split_size = 0
    split = {}

    while sizes:
        term, size = sizes.popitem()
        if size > max_size:
            print(f"Term {term} is too large: {size / 1024 / 1024:.2f} MB")
        if split_size + size >= max_size:
            save_split(split, index_dir, start_term, end_term)
            split_size, start_term, end_term, split = 0, None, None, {}
        split[term] = index.pop(term)
        split_size += size
        start_term = start_term or term
        end_term = term
    if split:
        save_split(split, index_dir, start_term, end_term)


def build_index(docs_file, index_dir):
    index = {}
    with open(docs_file, 'r') as file:
        for line in file:
            doc_id, _, title, *text = line.strip().split("\t")
            terms = preprocess(f"{title} {' '.join(text)}")
            for term in terms:
                index.setdefault(term, set()).add(doc_id)

    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    else:
        shutil.rmtree(index_dir)
        os.makedirs(index_dir)

    split_index(index, index_dir)


def load_terms(index_dir, terms):
    index = {}
    term_files = get_relevant_files(index_dir, terms)

    for term, files in term_files.items():
        for file in files:
            with open(os.path.join(index_dir, file)) as f:
                for line in f:
                    t, doc_ids = line.strip().split("\t")
                    if t == term:
                        index[term] = set(doc_ids.split())
                        break
    return index


def get_relevant_files(index_dir, terms):
    term_files = {term: [] for term in terms}
    for fname in os.listdir(index_dir):
        start, end = fname.split(".")[0].split("-")
        for term in terms:
            if start <= term <= end:
                term_files[term].append(fname)
    return term_files


def get_memory_usage():
    if platform.system() == "Windows":
        import ctypes
        process = ctypes.windll.kernel32.GetCurrentProcess()
        counters = ctypes.c_ulonglong()
        ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), ctypes.sizeof(counters))
        return counters.value / 1024 / 1024
    elif platform.system() in ("Linux", "Darwin"):
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    else:
        raise NotImplementedError("Unsupported platform")


def generate_submission(queries_file, objects_file, submission_file, index):
    queries = load_queries(queries_file)
    terms = set(term for query in queries.values() for term in query)
    # print(f"Memory usage: {get_memory_usage():.2f} MB")
    
    # print(f"Terms size: {calculate_object_size(terms) / 1024 / 1024:.2f} MB")
    # print(f"Queries size: {calculate_object_size(queries) / 1024 / 1024:.2f} MB")

    index = load_terms(index, terms)
    # print(f"Memory usage: {get_memory_usage():.2f} MB")
    # print(f"Index size: {calculate_object_size(index) / 1024 / 1024:.2f} MB")
    
    query_results = process_queries(queries, index)
    # print(f"Memory usage: {get_memory_usage():.2f} MB")
    
    write_submission_file(objects_file, submission_file, query_results)
    # print(f"Memory usage: {get_memory_usage():.2f} MB")


def load_queries(queries_file):
    queries = {}
    with open(queries_file, 'r') as file:
        for line in file:
            query_id, query = line.strip().split("\t")
            queries[query_id] = query.split(" ")
            # print(f"load_queries memory usage: {get_memory_usage():.2f} MB")
    return queries


def process_queries(queries, index):
    query_results = {}
    for query_id, query_terms in queries.items():
        sets = [index.get(term, set()) for term in query_terms]
        query_results[query_id] = set.intersection(*sets)
        # print(f"process_queries memory usage: {get_memory_usage():.2f} MB")
    return query_results


def write_submission_file(objects_file, submission_file, query_results):
    if os.path.exists(submission_file):
        os.remove(submission_file)

    with open(submission_file, "w") as submission_f:
        submission_f.write("ObjectId,Label\n")

        with open(objects_file, "r") as objects_f:
            next(objects_f)  # Skip header
            for line in objects_f:
                obj_id, query_id, doc_id = line.strip().split(",")
                label = int(doc_id in query_results.get(query_id, set()))
                submission_f.write(f"{obj_id},{label}\n")
                # print(f"write_submission_file memory usage: {get_memory_usage():.2f} MB")
    return query_results


def extract_zip_if_needed(zip_path, extract_to):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found at {zip_path}")
        
    if extract_to.exists() and extract_to.is_dir():
        print(f"Directory '{extract_to}' already exists. Skipping extraction.")
        return

    print(f"Extracting '{zip_path}' to '{extract_to}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction completed successfully.")
    

def main():
    parser = argparse.ArgumentParser(description='Indexing homework solution')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('--build_index', action='store_true', help='force reindexing')
    parser.add_argument('--index_dir', required=True, help='index directory')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()

    zip_path = Path(args.data_dir) / "boolean-retrieval-homework-vk-ir-fall-2024.zip"
    extract_to = Path(args.data_dir) / "boolean-retrieval-homework-vk-ir-fall-2024"

    # Ensure data is extracted
    extract_zip_if_needed(zip_path, extract_to)
    extract_to = extract_to / "boolean-retrieval-homework-vk-ir-fall-2024"
    if args.build_index:
        docs_file = extract_to / "vkmarco-docs.tsv"
        build_index(docs_file, args.index_dir)
    else:
        queries_file = extract_to / "vkmarco-doceval-queries.tsv"
        objects_file = extract_to / "objects.csv"
        
        generate_submission(queries_file, objects_file, args.submission_file, args.index_dir)


    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()