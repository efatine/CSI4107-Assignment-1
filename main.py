import json
import preprocessing
import indexing as idx
import ranking
import os

# file paths
stopwords_file = "stop_words.txt"
corpus_file = "scifact/corpus.jsonl"
queries_file = "scifact/queries.jsonl"

# 1. load stopwords
print("loading stopwords...")
stop_words = preprocessing.load_stopwords(stopwords_file)
print(f"loaded {len(stop_words)} stopwords.")

# 2. preprocess documents
print("loading and preprocessing corpus...")
documents = {} # map doc_id to list of tokens

try:
    with open(corpus_file, 'r', encoding='utf-8') as f:
        # read the file line by line because it is jsonl
        for line in f:
            data = json.loads(line)
            doc_id = data.get('_id')
            
            # combine title and text for better coverage
            full_text = data.get('title', '') + " " + data.get('text', '')
            
            # use the function from our preprocessing module
            tokens = preprocessing.preprocess_text(full_text, stop_words)
            
            # store the tokens
            documents[doc_id] = tokens
            
            # print the first document just to check
            if len(documents) == 1:
                print(f"sample doc {doc_id}: {tokens[:10]}...")

    print(f"finished preprocessing {len(documents)} documents")

except FileNotFoundError:
    print("corpus file not found")

# 3. preprocess queries
print("loading and preprocessing queries...")
processed_queries = []

try:
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            q_id = data.get('_id')
            
            # extracting the text
            text = data.get('text', '')
            
            # preprocess
            q_tokens = preprocessing.preprocess_text(text, stop_words)
            
            # store result
            processed_queries.append({'id': q_id, 'tokens': q_tokens})
            
            # print first query just to check 
            if len(processed_queries) == 1:
                print(f"sample query {q_id}: {q_tokens}")
                
    print(f"finished preprocessing {len(processed_queries)} queries.")

except FileNotFoundError:
    print("queries file not found. please check the path.")

# building inverted index
inverted_index = {}  #structure: {token: {doc_id: count, ...}, ...}
try:
    inverted_index = idx.build_inverted_index(documents) #processing structure
    print(f"inverted index built with {len(inverted_index)} unique tokens.")
    print(next(iter(inverted_index.items())))

    print("saving inverted index to file...")
    with open("inverted_index.json", "w") as f:  #save the dict to a json file
        json.dump(inverted_index, f, indent=2)
except Exception as e:
    print(f"error building inverted index: {e}")
    
  # --- Step 3: Retrieval + Ranking ---
print("converting inverted index format for ranking...")
postings = {term: list(doc_map.items()) for term, doc_map in inverted_index.items()}

print("computing df/idf + doc norms...")
df = ranking.compute_df(postings)
idf = ranking.compute_idf(df, len(documents))
doc_norm = ranking.compute_doc_norms(postings, idf)

# convert processed_queries list -> list[(qid_int, tokens)]
queries = []
for q in processed_queries:
    try:
        qid_int = int(q["id"])
    except:
        continue
    # OPTIONAL: only odd query ids (test set requirement)
    if qid_int % 2 == 0:
        continue
    queries.append((qid_int, q["tokens"]))

queries.sort(key=lambda x: x[0])

print("scoring and ranking...")
rankings = {}
for qid, q_tokens in queries:
    scores = ranking.score_query_tfidf_cosine(q_tokens, postings, idf, doc_norm)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
    rankings[qid] = top

print("writing Results...")

ranking.write_results("Results.txt", rankings, "tfidf_cosine_run")

print("Done.")