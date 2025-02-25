import torch
import transformers
import datasets
from datasets import load_dataset
from src.multi_task_pipeline import load_category_model, multi_task_pipeline
from src.utils.evaluation_metrics import evaluate_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Model ve Veri Seti Yükleme
category_model, tokenizer, labels = load_category_model()

dataset = load_dataset("ag_news", split="test").shuffle(seed=42).select(range(7600))  # Max 7600

# Model Testi (Inference) ve Karşılaştırma
y_true, y_pred = [], []
rouge_scores, bleu_scores = [], []

for i, example in enumerate(tqdm(dataset, desc="📊 İşleniyor", unit=" haber", dynamic_ncols=True, leave=True)):
    text, true_category = example["text"], labels[example["label"]]
    result = multi_task_pipeline(text, category_model, tokenizer, labels)

    y_true.append(true_category)
    y_pred.append(result["category"])

    # **Özetleme Metrikleri**
    summary_metrics = evaluate_metrics(text, result["summary"])
    rouge_scores.append(summary_metrics["ROUGE-L"])
    bleu_scores.append(summary_metrics["BLEU"])

# Model Performans Değerlendirmesi
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

# Özetleme metriklerinin ortalaması
avg_rouge = sum(rouge_scores) / len(rouge_scores)
avg_bleu = sum(bleu_scores) / len(bleu_scores)

print("\n### Genel Model Performansı ###")
print(f"Kategori Sınıflandırma - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
print(f"Özetleme - ROUGE-L: {avg_rouge:.4f}, BLEU: {avg_bleu:.4f}")
