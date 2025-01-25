import os
import pickle
import json
from pyserini.index.lucene import (
    # IndexReader,
    LuceneIndexReader,
    Document
)
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.search import (
    get_topics_with_reader,
)
from pyserini.search.lucene import LuceneSearcher
from pyserini.vectorizer import TfidfVectorizer
import numpy as np
from tqdm import tqdm
# from scipy.sparse.linalg import norm as sparse_norm
from scipy import sparse

# Load TSV-format topics
topic_file_path = "./files/queriesROBUST.txt"
index_path = f"./index/RobustPyserini"

topics = get_topics_with_reader(
    "io.anserini.search.topicreader.TsvIntTopicReader", topic_file_path
)
# fix query ids
queries = {}
for topic_id, topic in topics.items():
    fixed_topic_id = str(topic_id)
    if len(fixed_topic_id) == 2:
        fixed_topic_id = "0" + str(topic_id)
    queries[fixed_topic_id] = topic["title"]
print(len(queries))
assert len(queries) == 249, "missing queries"
# sort by topic id
queries = dict(sorted(queries.items()))


def rank_documents(run_number, method="bm25", stemmer="krovetz", top_k=1000):
    # Initialize the searcher with the path to your stemmed index
    searcher = LuceneSearcher(index_path)
    #vectorizer = TfidfVectorizer(index_path, min_df=3)
    # specify custom analyzer for the query processing step to match the way the index was built
    analyzer = get_lucene_analyzer(
        stemmer=stemmer, stopwords=False
    )  # Ensure no stopwords are removed from the query
    searcher.set_analyzer(analyzer)

    if method == "rm3":
        searcher.set_rm3(fb_terms=500, fb_docs=5, original_query_weight=0.3)
    elif method == "bm25":
        searcher.set_bm25(k1=0.9, b=0.4)
    elif method == "qld":
        searcher.set_qld(mu=1000)
    else:
        raise Exception("Invalid ranking method")

    # Loop through each query in the topics dictionary and retrieve documents:
    results = {}  # To store results for each query
    for topic_id, topic in queries.items():
        hits = searcher.search(
            topic, k=top_k
        )  # k=1000 is the number of retrieved documents
        # Store results in TREC format for each topic
        results[topic_id] = [(hit.docid, hit.lucene_docid, i+1, hit.score) for i, hit in enumerate(hits)]

    # Now you can save the results to a file in the TREC format:
    output_file = f"./results/run_{run_number}_{method}.res"
    if not os.path.exists("./results"):
        os.makedirs("./results")
    sorted_results = dict(sorted(results.items()))
    with open(output_file, "w") as f:
        for topic_id, hits in sorted_results.items():
            for rank, (docid, _, _, score) in enumerate(hits, start=1):
                f.write(f"{topic_id} Q0 {docid} {rank} {score:.4f} run{run_number}\n")


def rank_documents_vector(run_number, top_k=1000, stemmer="krovetz"):
    """
    Document ranking using sparse matrix operations and precomputations.
    """
    # Initialize index reader and get document IDs
    index_reader = None#LuceneIndexReader(index_path)
    num_docs = index_reader.stats()["documents"]
    all_docids = [
        index_reader.convert_internal_docid_to_collection_docid(i)
        for i in range(num_docs)
    ]
    # all_docids = all_docids[:1000]  # for testing purposes

    # Initialize vectorizer and load/create document matrix
    doc_vectorizer = TfidfVectorizer(lucene_index_path=index_path, verbose=True)
    doc_matrix_file = f"./results/doc_matrix_{run_number}.npz"
    doc_norms_file = f"./results/doc_norms_{run_number}.npy"

    if not os.path.exists(doc_matrix_file):
        # Build and save document matrix and norms
        print("Building document matrix...")
        doc_matrix = doc_vectorizer.get_vectors(all_docids)
        sparse.save_npz(doc_matrix_file, doc_matrix)

        # Precompute document norms
        print("Precomputing document norms...")
        doc_norms = np.sqrt(np.array(doc_matrix.power(2).sum(axis=1)).ravel())
        doc_norms[doc_norms == 0] = 1e-10  # Prevent division by zero
        np.save(doc_norms_file, doc_norms)
    else:
        # Load precomputed data
        print("Loading precomputed data...")
        doc_matrix = sparse.load_npz(doc_matrix_file)
        doc_norms = np.load(doc_norms_file)

    # Process queries
    results = {}
    query_list = list(queries.items())

    print("Retrieving documents for each query...")
    for topic_id, query_text in tqdm(query_list, desc="Queries", unit="query"):
        # Vectorize query
        query_vector = doc_vectorizer.get_query_vector(query_text)
        if query_vector.nnz == 0:  # Handle empty queries
            results[topic_id] = []
            continue

        query_sparse = query_vector
        # query_sparse = sparse.csr_matrix(query_vector)

        # cosine similarities
        qnorm = sparse.linalg.norm(query_sparse)
        if qnorm < 1e-10:
            similarity_scores = np.zeros(num_docs)
        else:
            # Matrix multiplication for all documents 
            scores = query_sparse.dot(doc_matrix.T).toarray().ravel()
            similarity_scores = scores / (qnorm * doc_norms)

        # top_k 
        ranked_indices = np.argsort(-similarity_scores)[:top_k]
        top_k_scores = similarity_scores[ranked_indices]
        top_k_docids = [all_docids[i] for i in ranked_indices]

        results[topic_id] = [
            (docid, rank, float(score))
            for rank, (docid, score) in enumerate(
                zip(top_k_docids, top_k_scores), start=1
            )
        ]

    # Save
    output_file = f"./results/run_{run_number}_vector.res"
    os.makedirs("./results", exist_ok=True)

    with open(output_file, "w") as fout:
        for topic_id in sorted(results.keys()):
            for rank, (docid, _, score) in enumerate(results[topic_id], start=1):
                fout.write(
                    f"{topic_id} Q0 {docid} {rank} {score:.4f} run{run_number}\n"
                )

    print(f"\nOptimized vector-based run file saved to {output_file}")


#rank_documents(run_number=1, method="rm3")
#rank_documents(run_number=10, method="bm25")
#rank_documents(run_number=3, method="qld")
#rank_documents_vector(run_number=5, top_k=1000, stemmer="krovetz")
