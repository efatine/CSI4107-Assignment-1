# ranking.py
# Step 3: Retrieval + Ranking using TF-IDF and cosine similarity

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from preprocessing import preprocess_text, load_stopwords


# reads jsonl file line by line
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# only use odd query IDs (test queries)
def is_odd_query_id(qid) -> bool:
    try:
        return int(qid) % 2 == 1
    except Exception:
        return True


# building inverted index
# structure: term -> list[(doc_id, tf)]
def build_inverted_index(
    corpus_path: str,
    stopwords_path: str,
    text_fields: List[str] = None,
) -> Tuple[Dict[str, List[Tuple[str, int]]], int]:

    stop_words = load_stopwords(stopwords_path)
    postings = defaultdict(list)

    N = 0  # total number of documents

    if text_fields is None:
        text_fields = ["title", "text", "abstract"]

    for doc in read_jsonl(corpus_path):
        doc_id = doc.get("_id") or doc.get("doc_id") or doc.get("id")
        if doc_id is None:
            continue

        # combine text fields
        parts = []
        for field in text_fields:
            val = doc.get(field)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())

        full_text = " ".join(parts)

        # preprocessing
        tokens = preprocess_text(full_text, stop_words)

        # compute term frequency for this doc
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        # add to postings list
        for term, freq in tf.items():
            postings[term].append((str(doc_id), int(freq)))

        N += 1

    return postings, N


# compute document frequency (df)
def compute_df(postings):
    return {term: len(plist) for term, plist in postings.items()}


# compute idf using log(N / df)
def compute_idf(df, N):
    idf = {}
    for term, dft in df.items():
        if dft > 0:
            idf[term] = math.log(N / dft)
    return idf


# compute document vector norms (for cosine similarity)
def compute_doc_norms(postings, idf):
    norm_sq = defaultdict(float)

    for term, plist in postings.items():
        if term not in idf:
            continue

        for doc_id, tf_td in plist:
            w = tf_td * idf[term]
            norm_sq[doc_id] += w * w

    return {doc_id: math.sqrt(v) for doc_id, v in norm_sq.items()}


# scoring function using TF-IDF + cosine similarity
def score_query_tfidf_cosine(query_tokens, postings, idf, doc_norm):

    # compute query term frequency
    qtf = defaultdict(int)
    for t in query_tokens:
        if t in postings and t in idf:
            qtf[t] += 1

    # compute query tf-idf weights
    wq = {}
    for t, tf_tq in qtf.items():
        wq[t] = tf_tq * idf[t]

    # compute query norm
    q_norm = math.sqrt(sum(w * w for w in wq.values()))
    if q_norm == 0:
        return {}

    dot = defaultdict(float)

    # compute dot product between query and docs
    for t, wqt in wq.items():
        for doc_id, tf_td in postings[t]:
            wdt = tf_td * idf[t]
            dot[doc_id] += wdt * wqt

    # final cosine scores
    scores = {}
    for doc_id, dp in dot.items():
        dn = doc_norm.get(doc_id, 0.0)
        if dn > 0:
            scores[doc_id] = dp / (dn * q_norm)

    return scores


# load and preprocess queries
def read_queries(queries_path, stopwords_path, mode):

    stop_words = load_stopwords(stopwords_path)
    out = []

    for q in read_jsonl(queries_path):
        qid = q.get("_id") or q.get("id") or q.get("query_id")
        if qid is None:
            continue

        # only odd queries (test)
        if not is_odd_query_id(qid):
            continue

        title = q.get("title") or ""
        text = q.get("text") or ""

        # choose query representation
        if mode == "title":
            raw = title.strip() if title.strip() else text.strip()
        else:
            raw = (title.strip() + " " + text.strip()).strip()

        tokens = preprocess_text(raw, stop_words)

        try:
            qid_int = int(qid)
        except Exception:
            continue

        out.append((qid_int, tokens))

    out.sort(key=lambda x: x[0])
    return out


# write results in required TREC format
def write_results(path, rankings, run_name):

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(rankings.keys()):
            for rank, (doc_id, score) in enumerate(rankings[qid], start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


# main execution
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="scifact/corpus.jsonl")
    ap.add_argument("--queries", default="scifact/queries.jsonl")
    ap.add_argument("--stopwords", default="stop_words.txt")
    ap.add_argument("--mode", choices=["title", "titletext"], default="title")
    ap.add_argument("--output", default="Results")
    ap.add_argument("--run-name", default="tfidf_cosine")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    # Step 1: build inverted index
    print("Building inverted index...")
    postings, N = build_inverted_index(args.corpus, args.stopwords)
    print(f"N={N}, vocab={len(postings)}")

    # Step 2: compute df, idf, and doc norms
    df = compute_df(postings)
    idf = compute_idf(df, N)
    doc_norm = compute_doc_norms(postings, idf)

    # Step 3: read queries
    queries = read_queries(args.queries, args.stopwords, args.mode)

    # Step 4: score and rank
    rankings = {}
    for qid, q_tokens in queries:
        scores = score_query_tfidf_cosine(q_tokens, postings, idf, doc_norm)
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: args.topk]
        rankings[qid] = top

    # Step 5: write results
    write_results(args.output, rankings, args.run_name)

    print("Done.")


if __name__ == "__main__":
    main()
