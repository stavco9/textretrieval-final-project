import os
from pyserini.index.lucene import IndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.search import get_topics_with_reader, LuceneSearcher

# Load TSV-format topics
topic_file_path = "./files/queriesROBUST.txt"
index_path = f'./index/RobustPyserini'

topics=get_topics_with_reader('io.anserini.search.topicreader.TsvIntTopicReader',topic_file_path)
#fix query ids
queries = {}
for topic_id, topic in topics.items():
    fixed_topic_id = str(topic_id)
    if len(fixed_topic_id) == 2:
        fixed_topic_id = '0'+str(topic_id)
    queries[fixed_topic_id] = topic['title']
print(len(queries))
assert len(queries) == 249, 'missing queries'

def rank_documents(run_number, method="bm25", stemmer="krovetz", top_k=1000):
    # Initialize the searcher with the path to your stemmed index
    searcher = LuceneSearcher(index_path)
    # specify custom analyzer for the query processing step to match the way the index was built
    analyzer = get_lucene_analyzer(stemmer=stemmer, stopwords=False) #Ensure no stopwords are removed from the query
    searcher.set_analyzer(analyzer)

    if method == "rm3":
        searcher.set_rm3(fb_terms=100, fb_docs=3, original_query_weight=0.3)
    elif method == "bm25":
        searcher.set_bm25(k1=0.9, b=0.4)
    elif method == "qld":
        searcher.set_qld(mu=1000)
    else:
        raise Exception("Invalid ranking method")

    #Loop through each query in the topics dictionary and retrieve documents:
    results = {} # To store results for each query
    for topic_id, topic in queries.items():
        hits = searcher.search(topic,k=top_k) # k=1000 is the number of retrieved documents
        # Store results in TREC format for each topic
        results[topic_id] = [(hit.docid, i+1, hit.score) for i, hit in enumerate(hits)]
    
    # Now you can save the results to a file in the TREC format:
    output_file = f'./results/run_{run_number}_{method}.res'
    if not os.path.exists('./results'):
        os.makedirs('./results')
    sorted_results = dict(sorted(results.items()))
    with open(output_file, 'w') as f:
        for topic_id, hits in sorted_results.items():
            for rank, (docid, _, score) in enumerate(hits, start=1):
                f.write(f"{topic_id} Q0 {docid} {rank} {score:.4f} run{run_number}\n")

rank_documents(1, 'rm3')
#rank_documents(2, 'bm25')
#rank_documents(3, 'qld')