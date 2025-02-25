import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import yaml
import os
from sklearn.metrics import accuracy_score

# **1️⃣ Config Dosyasını Yükle**
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# **2️⃣ Veri Setini Yükle (AG News)**
dataset = load_dataset("ag_news")

# **3️⃣ Model ve Tokenizer Tanımla**
model_name = config["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)

# **4️⃣ Veri Setini Ön İşle**
from utils.preprocessing import preprocess_function

tokenized_datasets = dataset.map(
    lambda examples: preprocess_function(examples, tokenizer),  # Tokenizer'ı ilet
    batched=True
)

# **5️⃣ Modeli Yükle**
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=config["num_labels"]
)

# **6️⃣ compute_metrics Fonksiyonunu Tanımla**
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"eval_accuracy": accuracy}

# **7️⃣ Eğitim Parametreleri Güncellendi**
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    evaluation_strategy=config["evaluation_strategy"],
    save_strategy=config["save_strategy"],
    learning_rate=float(config["learning_rate"]),
    per_device_train_batch_size=config["train_batch_size"],
    per_device_eval_batch_size=config["eval_batch_size"],
    num_train_epochs=config["num_train_epochs"],
    weight_decay=config["weight_decay"],
    logging_dir=config["logging_dir"],
    lr_scheduler_type="linear",
    warmup_ratio=config["warmup_ratio"],
    save_total_limit=config["save_total_limit"],
    load_best_model_at_end=config["load_best_model_at_end"],
    metric_for_best_model=config["metric_for_best_model"],
    greater_is_better=config["greater_is_better"],
    fp16=config.get("fp16", False),
    bf16=config.get("bf16", False),
    gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
)

# **8️⃣ Trainer Nesnesi**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,  # compute_metrics fonksiyonunu ilet
)

# **9️⃣ Modeli Eğit ve Checkpoint Kullan**
checkpoint_dir = os.path.join(config["output_dir"], "checkpoint-last")
if os.path.exists(checkpoint_dir):
    print(f"Checkpoint bulundu! Eğitime {checkpoint_dir} konumundan devam ediliyor...")
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    trainer.train()

# **🔟 Eğitilmiş Modeli Kaydet**
model_save_path = config["model_save_path"]
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save_pretrained(os.path.dirname(model_save_path))
tokenizer.save_pretrained(os.path.dirname(model_save_path))

print(f"Eğitim tamamlandı! Model {model_save_path} dizinine kaydedildi.")