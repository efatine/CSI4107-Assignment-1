# CSI 4107 - Assignment 1

## Group Information & Task Division

- **Vishal Bhat (300247928):**
  - (Step 3) Retrieval and Ranking
- **Kevin Xiong (300236877):**
  - (Step 3) Retrieval and Ranking
- **Elias Fatine (300197450):**
  - (Step 1) Preprocessing
  - Report Writing
- **Carl Li (300235679):**
  - (Step 2) Indexing
  - Report Writing

---

## Program Description

### Step 1: Preprocessing

For the first step, we needed to clean up the text from the documents and the queries so the indexing would work better. Our code uses regular expressions (regex) to strip out anything that isn't a letter and converts everything to lowercase. We also loaded the `stop_words.txt` file to remove common words that don't help with searching. We also decided to remove any words that were 2 letters or shorter to get rid of extra noise. Also, we combined the Title and Text (abstract) together so that we wouldn't miss important words that might not be in the title

### Step 2: Indexing

We built the inverted index using a nested dictionary structure. It looks something like this:

```json
{
  "token": {"doc_id": count, ...},
  ...
}

```

We saved this index to a file called `inverted_index.json`. This was helpful because it meant we didn't have to rebuild the index every time we wanted to test the program

### Step 3: Retrieval and Ranking

To rank the documents, we implemented the TF-IDF weights and Cosine Similarity formulas from the lecture slides:

- **Term Frequency (TF):** We normalized this by dividing the count by the max frequency in the document so long documents wouldn't unfairly get higher scores
- **Inverse Document Frequency (IDF):** We calculated this using the formula that we saw in class

---

## How to Run It

1.  Make sure you have Python 3 installed
2.  Put all the files in a folder like this (the structure needs to be the same):
    ```text
    /assignment_folder
       main.py
       preprocessing.py
       indexing.py
       ranking.py
       stop_words.txt
       /scifact
           corpus.jsonl
           queries.jsonl
           /qrels
               test.tsv
    ```
3.  Open your terminal in that folder
4.  Run this command:
    ```bash
    python main.py
    ```
5.  When it runs, it will:
    - Print the vocabulary size to the console
    - Save 100 random tokens to a file called `sample_tokens.txt`
    - Print the MAP score right in the console
    - Save the final ranked list to a file called `Results`

---

## Algorithms & Data Structures

### Data Structures

- **Inverted Index:** We used a Python Dictionary (`{word: {doc_id: count}}`). Looking up words in a dictionary takes (O(1)) time, so it makes the search efficient
- **Max Frequency Map:** We created a separate dictionary to store the highest word count for every document. We needed this to normalize the term frequency properly

### Algorithms

- **Vector Space Model:** We treated the queries and documents like vectors
- **TF-IDF Weighting:** We used the formulas from the lecture slides:
  - **TF:** Normalized by dividing the raw count by the max frequency in that doc ($freq / max\_freq$).
  - **IDF:** Calculated as $\log_2(N / df)$
- **Cosine Similarity:** To rank them, we calculated the dot product of the query vector and the document vector

### Optimizations

- **Full Text Indexing:** Initially, we tried just using the titles, but the results weren't great. We switched to combining the Title + Abstract, which improved the results a lot because the queries are very specific
- **Sparse Search:** In our ranking loop, we only look at documents that actually contain the query terms. We don't waste time checking documents that have nothing to do with the query

### Vocabulary Info

- **Vocabulary Size:** 29,542 tokens
- **Sample of 100 Tokens:**
  ```text
  ['microstructural', 'development', 'human', 'newborn', 'cerebral', 'white', 'matter', 'assessed',
  'vivo', 'diffusion', 'tensor', 'magnetic', 'resonance', 'imaging', 'alterations', 'architecture',
  'developing', 'brain', 'affect', 'cortical', 'result', 'functional', 'disabilities', 'line', 'scan',
  'weighted', 'mri', 'sequence', 'analysis', 'applied', 'measure', 'apparent', 'coefficient',
  'calculate', 'relative', 'anisotropy', 'delineate', 'dimensional', 'fiber', 'preterm', 'full',
  'term', 'infants', 'assess', 'effects', 'prematurity', 'early', 'gestation', 'studied', 'central',
  'mean', 'microm', 'decreased', 'posterior', 'limb', 'internal', 'capsule', 'coefficients', 'versus',
  'closer', 'birth', 'absolute', 'values', 'areas', 'compared', 'nonmyelinated', 'fibers', 'corpus',
  'callosum', 'visible', 'marked', 'differences', 'organization', 'data', 'indicate', 'quantitative',
  'assessment', 'water', 'insight', 'living', 'induction', 'myelodysplasia', 'myeloid', 'derived',
  'suppressor', 'cells', 'myelodysplastic', 'syndromes', 'mds', 'age', 'dependent', 'stem', 'cell',
  'malignancies', 'share', 'biological', 'features', 'activated', 'adaptive', 'immune']
  ```

---

## Results & Discussion

### MAP Score

We got a Mean Average Precision (MAP) of **0.4857**

### Discussion

- **Title vs Full Text:** As I mentioned in the optimizations, using the full text (Title + Abstract) worked way better than just the Title. Since the SciFact dataset is about scientific claims, you really need the details in the abstract to find a match
- **Stemming:** We didn't use a stemmer (like Porter Stemmer), we just did exact matching. The assignment instructions did say that stemming might help, and I think it would. For example, right now "cell" and "cells" are treated as totally different words. If we combined them, our score would probably go up. But even without it, getting a MAP of almost 0.49 seems pretty good for this implementation

### First 10 Answers (Sample Output)

### Query 1

```text
1 Q0 13231899 1 0.085769 my_vsm_run
1 Q0 10931595 2 0.074656 my_vsm_run
1 Q0 10608397 3 0.074027 my_vsm_run
1 Q0 10607877 4 0.072092 my_vsm_run
1 Q0 24998637 5 0.071426 my_vsm_run
1 Q0 31543713 6 0.064316 my_vsm_run
1 Q0 25404036 7 0.053258 my_vsm_run
1 Q0 9580772 8 0.052053 my_vsm_run
1 Q0 16939583 9 0.050176 my_vsm_run
1 Q0 6863070 10 0.049409 my_vsm_run

```

### Query 2

```text
3 Q0 23389795 1 0.348582 my_vsm_run
3 Q0 2739854 2 0.310383 my_vsm_run
3 Q0 14717500 3 0.243748 my_vsm_run
3 Q0 4632921 4 0.212910 my_vsm_run
3 Q0 8411251 5 0.189150 my_vsm_run
3 Q0 4378885 6 0.148054 my_vsm_run
3 Q0 32181055 7 0.142827 my_vsm_run
3 Q0 3672261 8 0.140804 my_vsm_run
3 Q0 14019636 9 0.136908 my_vsm_run
3 Q0 4414547 10 0.132784 my_vsm_run
```
