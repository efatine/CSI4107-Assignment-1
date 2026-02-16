import json
import math
import preprocessing
import indexing as idx
import ranking

# file paths
corpus_file = "scifact/corpus.jsonl"
queries_file = "scifact/queries.jsonl"
qrels_file = "scifact/qrels/test.tsv"
stopwords_file = "stop_words.txt"
output_file = "Results"
sample_tokens_file = "sample_tokens.txt"
run_name = "my_vsm_run"

# function to load qrels for map score
def load_qrels(filepath):
    relevant_docs = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] == 'query-id': continue
                
                # check if tsv or trec format
                if len(parts) == 3: qid, did, score = parts[0], parts[1], parts[2]
                elif len(parts) >= 4: qid, did, score = parts[0], parts[2], parts[3]
                else: continue
                
                try:
                    if float(score) > 0:
                        if qid not in relevant_docs: relevant_docs[qid] = set()
                        relevant_docs[qid].add(did)
                except ValueError: continue
    except FileNotFoundError:
        print(f"warning: qrels file not found at {filepath}")
    return relevant_docs

# calculate average precision for one query
def calculate_average_precision(retrieved_docs, relevant_docs_set):
    if not relevant_docs_set: return 0.0
    hits = 0
    sum_precisions = 0
    for i, (doc_id, score) in enumerate(retrieved_docs, 1):
        if doc_id in relevant_docs_set:
            hits += 1
            sum_precisions += (hits / i)
    return sum_precisions / len(relevant_docs_set)

# --- main execution ---

# 1. load stopwords
print("loading stopwords...")
stop_words = preprocessing.load_stopwords(stopwords_file)
print(f"loaded {len(stop_words)} stopwords")

# 2. load and preprocess corpus
print("loading and preprocessing corpus...")
documents = {} 
doc_max_freqs = {} # need this for tf calculation

try:
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = data.get('_id')
            
            # combine title and text
            full_text = data.get('title', '') + " " + data.get('text', '')
            
            # preprocess text
            tokens = preprocessing.preprocess_text(full_text, stop_words)
            documents[doc_id] = tokens
            
            # calculate max freq for this doc
            max_f = 0
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
                if counts[t] > max_f: max_f = counts[t]
            doc_max_freqs[str(doc_id)] = max_f if max_f > 0 else 1

    print(f"finished preprocessing {len(documents)} documents")
except FileNotFoundError:
    print(f"error: corpus file {corpus_file} not found")
    exit()

# 3. build inverted index
print("building inverted index...")
inverted_index = idx.build_inverted_index(documents)

# report vocab size and sample tokens
vocabulary = list(inverted_index.keys())
print("--------------------------------------------------")
print(f"Vocabulary size: {len(vocabulary)} tokens")
print("--------------------------------------------------")
sample_100 = vocabulary[:100]
print("Sample of 100 tokens (saved to file):")
print(sample_100)
print("--------------------------------------------------")
with open(sample_tokens_file, "w", encoding="utf-8") as f:
    f.write(str(sample_100))

# 4. compute weights
print("Computing tf-idf weights...")
postings = inverted_index 
N = len(documents)
df = ranking.compute_df(postings)
idf = ranking.compute_idf(df, N)
doc_norm = ranking.compute_doc_norms(postings, idf, doc_max_freqs)

# 5. process queries
print("Processing queries...")
processed_queries = []
try:
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            q_id = data.get('_id')
            
            # only test queries (odd ids)
            if int(q_id) % 2 == 0: continue
            
            # use title and text
            text = (data.get('title', '') + " " + data.get('text', '')).strip()
            q_tokens = preprocessing.preprocess_text(text, stop_words)
            processed_queries.append((q_id, q_tokens))
except FileNotFoundError:
    print("queries file not found")

# 6. ranking
print("ranking queries...")
all_rankings = {}

for q_id, q_tokens in processed_queries:
    scores = ranking.score_query_tfidf_cosine(q_tokens, postings, idf, doc_norm, doc_max_freqs)
    
    # keep top 100
    top_100 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
    all_rankings[str(q_id)] = top_100

# 7. write results
print(f"writing results to {output_file}...")
ranking.write_results(output_file, all_rankings, run_name)

# print first 10 for report
print("\nFirst 10 answers (for report):")
print("--------------------------------------------------")
with open(output_file, 'r') as f:
    for i, line in enumerate(f):
        if i < 10: print(line.strip())
print("--------------------------------------------------")

# 8. calculate map
print("\nCalculating MAP...")
qrels = load_qrels(qrels_file)
if qrels:
    total_ap = 0
    num_queries = 0
    for qid in all_rankings:
        if qid in qrels:
            retrieved = all_rankings[qid] 
            ap = calculate_average_precision(retrieved, qrels[qid])
            total_ap += ap
            num_queries += 1
    
    map_score = total_ap / num_queries if num_queries > 0 else 0
    print(f"mean average precision (MAP): {map_score:.4f}")
else:
    print("noNE found")

print("done")