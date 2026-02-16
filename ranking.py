# ranking.py
# Step 3: Retrieval + Ranking using TF-IDF and cosine similarity

import math
import os
from collections import defaultdict

# compute document frequency (df)
def compute_df(postings):
    return {term: len(plist) for term, plist in postings.items()}

# compute idf using log(N / df)
def compute_idf(df, N):
    idf = {}
    for term, dft in df.items():
        if dft > 0:
            idf[term] = math.log(N / dft, 2)
    return idf

# compute document vector norms (for cosine similarity)
def compute_doc_norms(postings, idf, doc_max_freqs):
    norm_sq = defaultdict(float)

    for term, plist in postings.items():
        if term not in idf:
            continue
        
        # check if plist is a dict or list to handle indexing format
        iterator = plist.items() if isinstance(plist, dict) else plist

        for doc_id, count in iterator:
            # normalize tf using max freq
            max_f = doc_max_freqs.get(str(doc_id), 1)
            norm_tf = count / max_f
            
            w = norm_tf * idf[term]
            norm_sq[str(doc_id)] += w * w

    return {doc_id: math.sqrt(v) for doc_id, v in norm_sq.items()}

# scoring function using TF-IDF + cosine similarity
def score_query_tfidf_cosine(query_tokens, postings, idf, doc_norm, doc_max_freqs):

    # compute query term frequency
    qtf = defaultdict(int)
    max_q_tf = 0
    for t in query_tokens:
        if t in postings and t in idf:
            qtf[t] += 1
            if qtf[t] > max_q_tf:
                max_q_tf = qtf[t]
    
    if max_q_tf == 0:
        return {}

    # compute query tf-idf weights
    wq = {}
    q_norm_sq = 0
    for t, tf_tq in qtf.items():
        norm_tf = tf_tq / max_q_tf
        w = norm_tf * idf[t]
        wq[t] = w
        q_norm_sq += w * w

    # compute query norm
    q_norm = math.sqrt(q_norm_sq)
    if q_norm == 0:
        return {}

    dot = defaultdict(float)

    # compute dot product between query and docs
    for t, wqt in wq.items():
        if t in postings:
            # check iteration format again
            iterator = postings[t].items() if isinstance(postings[t], dict) else postings[t]

            for doc_id, tf_td in iterator:
                # normalize doc tf
                max_d_tf = doc_max_freqs.get(str(doc_id), 1)
                norm_tf_d = tf_td / max_d_tf
                
                wdt = norm_tf_d * idf[t]
                dot[str(doc_id)] += wdt * wqt

    # final cosine scores
    scores = {}
    for doc_id, dp in dot.items():
        dn = doc_norm.get(doc_id, 0.0)
        if dn > 0:
            scores[doc_id] = dp / (dn * q_norm)

    return scores

# write results in required TREC format
def write_results(path, rankings, run_name):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        # sort query ids
        try:
            sorted_qids = sorted(rankings.keys(), key=lambda x: int(x))
        except:
            sorted_qids = sorted(rankings.keys())
            
        for qid in sorted_qids:
            for rank, (doc_id, score) in enumerate(rankings[qid], start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")