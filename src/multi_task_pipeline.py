from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.utils.keyword_extraction import extract_keywords
from src.utils.ner_extraction import extract_dates_locations
from src.utils.text_summarization import summarize_text
from src.utils.zero_shot_classification import classify_zero_shot, load_zero_shot_model
import torch
import uuid
import torch.nn.utils.prune as prune
import os


def load_category_model():
    """Eğitilmiş haber sınıflandırma modelini yükler."""
    model_path = os.path.join(os.path.dirname(__file__), "models/ag_news_classifier")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    prune_amount = 0.1

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_amount)
            prune.remove(module, "weight")
    return model, tokenizer, labels

def multi_task_pipeline(text, category_model, tokenizer, labels, use_zero_shot=False, zero_shot_model=None, zero_shot_tokenizer=None, custom_labels=None):
    """Verilen haber metnini analiz eder ve tüm görevleri uygular."""
    # Haber için rastgele bir UUID oluştur
    news_id = str(uuid.uuid4())

    if use_zero_shot and custom_labels:
        zero_shot_result = classify_zero_shot(text, custom_labels.split(","), zero_shot_model, zero_shot_tokenizer)
        category = zero_shot_result["label"]
        confidence = zero_shot_result["confidence"]
    else:
        # Kategori sınıflandırması
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = category_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
        confidence = torch.softmax(logits, dim=1).max().item()
        category = labels[predicted_class]

    # Anahtar kelime çıkarımı
    keywords = extract_keywords(text)

    # Varlık tanıma (Tarih ve lokasyon çıkarımı)
    extracted_entities = extract_dates_locations(text)
    locations = extracted_entities.get("locations", [])
    dates = extracted_entities.get("dates", [])

    # Metin özetleme
    summary = summarize_text(text)

    # Confidence Score Hesaplama (Kategori + Çıkarım)
    extraction_confidence = (confidence + 0.95) / 2

    return {
        "news_id": news_id,
        "category": category,
        "keywords": keywords,
        "entities": {
            "locations": locations,
            "dates": dates
        },
        "summary": summary,
        "confidence_scores": {
            "category": confidence,
            "extraction": extraction_confidence
        }
    }