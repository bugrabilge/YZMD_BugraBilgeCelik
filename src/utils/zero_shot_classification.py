from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def load_zero_shot_model(model_name="facebook/bart-large-mnli"):
    """Zero-Shot sınıflandırma modeli yükler."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def classify_zero_shot(text, candidate_labels, model, tokenizer):
    """Zero-shot sınıflandırma yapar."""
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
    result = classifier(text, candidate_labels)
    return {"label": result["labels"][0], "confidence": result["scores"][0]}
