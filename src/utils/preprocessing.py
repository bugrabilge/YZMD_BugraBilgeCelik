import nltk
from nltk.corpus import stopwords
import re
import spacy
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def preprocess_function(examples, tokenizer):  # Tokenizer parametre olarak eklendi
    texts = []
    labels = examples.get("label", examples.get("labels", None))

    for text in examples["text"]:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join([word for word in text.split() if word not in stop_words])
        text = " ".join([token.lemma_ for token in nlp(text) if not token.is_punct])
        texts.append(text)

    tokenized_inputs = tokenizer(texts, padding="max_length", truncation=True)

    if labels is None:
        labels = [0] * len(texts)
    else:
        labels = [int(label) for label in labels]

    tokenized_inputs["labels"] = labels
    return tokenized_inputs