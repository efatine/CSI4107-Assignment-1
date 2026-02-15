def build_inverted_index(documents):
    try:
        invertindex = {}
        for doc_id, tokens in documents.items():
            for token in tokens:
                if token not in invertindex: #if token is not in the index, add it with the current doc_id and count 1
                    invertindex[token] = {doc_id: 1}
                elif doc_id not in invertindex[token]: #if token is in the index but not for this doc_id, add the doc_id with count 1
                    invertindex[token][doc_id] = 1
                else:  # if token is in the index and for this doc_id, increment the count
                    invertindex[token][doc_id] += 1
    except Exception as e:
        print(f"error building inverted index: {e}")
    return invertindex

