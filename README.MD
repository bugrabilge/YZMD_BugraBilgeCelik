# Multi-Task NLP Pipeline

Bu proje, AG News veri seti üzerinde eğitilmiş bir model ile haber metinlerini analiz eder ve aşağıdaki görevleri gerçekleştirir:
- **Kategori sınıflandırma** (World, Sports, Business, Sci/Tech)
- **Anahtar kelime çıkarımı**
- **Tarih ve lokasyon tespiti**
- **Metin özetleme**
- **Zero-Shot sınıflandırma** (Opsiyonel olarak farklı kategorilerde değerlendirme)
- **İngilizce - Almanca Desteği**

## 🚀 Kurulum Adımları

### 1️⃣ Virtual Environment Oluşturma

```bash
python3 -m venv venv
source venv/bin/activate  # MacOS/Linux
dot\venv\Scripts\activate  # Windows
```

### 2️⃣ Gerekli Paketlerin Kurulumu

```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_trf
```

### 3️⃣ Model Checkpoint'lerini İndirme

Eğer AG News sınıflandırma modeli yerel olarak bulunmuyorsa, aşağıdaki komutu çalıştırarak indirilebilir:
```bash
mkdir -p src/models && cd src/models
git clone https://huggingface.co/username/ag_news_classifier
```

---

## 📊 Veri Seti Hazırlama

### 1️⃣ AG News Dataset'ini İndirme

```bash
from datasets import load_dataset
dataset = load_dataset("ag_news")
```

### 2️⃣ Ön İşleme Pipeline'ını Çalıştırma
```bash
python3 src/preprocessing.py
```

---

## 🎯 Model Eğitimi

### 1️⃣ Eğitim Komutları

```bash
python3 src/train.py
```

Hyperparameter ayarlarını `config/config.yaml` dosyasından değiştirebilirsiniz.

### 2️⃣ Multi-GPU Desteği

Modeli birden fazla GPU'da çalıştırmak için:
```bash
CUDA_VISIBLE_DEVICES=0,1 python src/train.py --multi-gpu
```

### 3️⃣ Checkpoint Kaydetme/Yükleme

Eğitim sırasında en iyi modeli kaydetmek için `config.yaml` dosyasında şu ayarı açabilirsiniz:
```yaml
load_best_model_at_end: true
save_total_limit: 2
```

Eğitime kaldığınız yerden devam etmek için:
```bash
python3 src/train.py --resume_from_checkpoint
```

---

## 🔍 Inference (Tahmin) Alma

### 1️⃣ Tekli Haber Analizi

```bash
python3 src/multi_task_pipeline.py --text "NASA launches a new satellite to explore Mars."
```

### 2️⃣ Batch Inference

```bash
python3 src/inference.py
```

### 3️⃣ API Kullanımı

```bash
python3 src/multi_task_web_demo.py
```
Yerel sunucu başlatıldıktan sonra aşağıdaki adres üzerinden erişebilirsiniz:
```
http://127.0.0.1:7860
```

---

## 📈 Model Değerlendirme

### 1️⃣ Metrik Hesaplama

```bash
python3 src/inference.py
```

Çıktı örneği:
```
Kategori Sınıflandırma - Accuracy: 0.86, Precision: 0.87, Recall: 0.86, F1-score: 0.86
Özetleme - ROUGE-L: 0.49, BLEU: 0.49
```

### 2️⃣ Test Seti Değerlendirmesi

AG News test setiyle model değerlendirmesi yapmak için:
```bash
python3 notebooks/model_development.ipynb
```

---

## ⚠️ Troubleshooting (Hata Giderme)

- **Modül Bulunamıyor Hatası**: `ModuleNotFoundError: No module named 'utils'`
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    ```
- **CUDA Hata Veriyor**: Eğer PyTorch CUDA ile çalışmıyorsa, aşağıdaki komut ile kontrol edebilirsiniz:
    ```python
    import torch
    print(torch.cuda.is_available())
    ```
- **Web Arayüzü Çalışmıyor**: `gradio`'nun kurulu olduğundan emin olun:
    ```bash
    pip install gradio
    ```

---

## ✅ Best Practices ve Öneriler

- **Büyük Modeller için GPU Kullanımı**: Modelinizi CUDA destekli bir GPU'da çalıştırmak için `torch.device("cuda")` kullanın.
- **Daha İyi Özetleme Performansı İçin Beam Search Kullanımı**: `num_beams=20, length_penalty=1.5` ayarları önerilir.
- **Zero-Shot için Özel Etiket Seti Kullanımı**: Kendi etiketlerinizi `custom_labels="Finance, Politics, Technology"` şeklinde girerek Zero-Shot sınıflandırma yapabilirsiniz.

---


