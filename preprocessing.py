import re

# load the stopwords from the file
def load_stopwords(filepath):
    stop_words = set()
    try:
        # read lines
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            # simple regex to find words inside the html tags if it's html
            # or just split by newlines if it is a list
            words = re.findall(r'[a-zA-Z]+', text)
            for w in words:
                stop_words.add(w.lower())
    except FileNotFoundError:
        # file is missing
        print("stopwords file not found")
    return stop_words

# tokenizer that removes punctuation and numbers
def tokenize(text):
    # remove non alphabetic characters using regex
    # keep only letters and whitespace
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # convert to lower case
    text = text.lower()
    # split by whitespace
    tokens = text.split()
    return tokens

# function to filter out stopwords
def remove_stopwords(tokens, stop_words):
    filtered_tokens = []
    for t in tokens:
        if t not in stop_words and len(t) > 2:
            # only keep words longer than 2 chars
            filtered_tokens.append(t)
    return filtered_tokens

# main preprocess function
def preprocess_text(text, stop_words):
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, stop_words)
    return tokens