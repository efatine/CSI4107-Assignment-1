# CSI 4107 - Assignment 1

Group information & Task Division:
    Vishal Bhat 300247928:
        (Step3) Retrieval and Ranking
    Kevin Xiong 300236877:
        (Step3) Retrieval and Ranking
    Elias Fatine 300197450:
        (Step1) Preprocessing
        Report Writing
    Carl Li 300235679
        (Step2) Indexing


Program Description:
    Step 1:
        In this step, the goal is to clean and normalize the text from both the corpus and the queries to ensure high-quality indexing. the system uses egular expressions to remove all non-alphabetic characters, and convert the tokens to lowercase. Then it loads "stop_words.txt", compares with the tokens and discards them. After that, all tokens with a length greater than 2 will be removed to reduce residual noises, and finally we combined the title and text fields to provide a comprehensive representation of each scientific document.

    Step 2:
        the system uses a 2D dictionary to contain the inverted index. the structure shows below:
            {token: {doc_id: count, ...}, ...}
        The built index is serialized and saved as "inverted_index.json". This allows the system to reload the index without re-processing the entire corpus in future runs.
        Vocabulary size: 29542 tokens
        Sample of 100 tokens (saved to file):
            ['microstructural', 'development', 'human', 'newborn', 'cerebral', 'white', 'matter', 'assessed', 'vivo', 'diffusion', 'tensor', 'magnetic', 'resonance', 'imaging', 'alterations', 'architecture', 'developing', 'brain', 'affect', 'cortical', 'result', 'functional', 'disabilities', 'line', 'scan', 'weighted', 'mri', 'sequence', 'analysis', 'applied', 'measure', 'apparent', 'coefficient', 'calculate', 'relative', 'anisotropy', 'delineate', 'dimensional', 'fiber', 'preterm', 'full', 'term', 'infants', 'assess', 'effects', 'prematurity', 'early', 'gestation', 'studied', 'central', 'mean', 'microm', 'decreased', 'posterior', 'limb', 'internal', 'capsule', 'coefficients', 'versus', 'closer', 'birth', 'absolute', 'values', 'areas', 'compared', 'nonmyelinated', 'fibers', 'corpus', 'callosum', 'visible', 'marked', 'differences', 'organization', 'data', 'indicate', 'quantitative', 'assessment', 'water', 'insight', 'living', 'induction', 'myelodysplasia', 'myeloid', 'derived', 'suppressor', 'cells', 'myelodysplastic', 'syndromes', 'mds', 'age', 'dependent', 'stem', 'cell', 'malignancies', 'share', 'biological', 'features', 'activated', 'adaptive', 'immune']
        
    Step 3:
        The system ranks documents using the TF-IDF weighting scheme and Cosine Similarity:
            Term Frequency: Normalized by the maximum frequency in the document to prevent bias.
            Inverse Document Frequency: Calculated as log2(Total_Docs / Docs_with_Term).

    mean average precision (MAP): 0.4857

    Sample Results:
        Q1:
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
        Q3:
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

    Discussion:
        Our system uses both Title and Text. In our experiments, this approach significantly outperformed the Title-only method. This is because scientific queries in Scifact are often highly specific, and the core evidence required for a match is typically found within the Text rather than the brief Title.