import json
import preprocessing

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