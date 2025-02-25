from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

MODEL_NAME = "google/pegasus-cnn_dailymail"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def summarize_text(text, max_words=50):
    """Verilen metni en fazla 50 kelimeye kadar özetler."""
    input_length = len(text.split())
    max_words = min(input_length - 5, 50)
    min_words = min(3, max_words // 3)
    summary = summarizer(
        text,
        max_length=max_words,
        min_length=min_words,
        do_sample=False,
        num_beams=20,
        length_penalty=1.3,
        repetition_penalty=1.4,
        num_return_sequences=1,
        no_repeat_ngram_size=3
    )
    summary_text = summary[0]["summary_text"].replace("<n>", "")
    return summary_text
