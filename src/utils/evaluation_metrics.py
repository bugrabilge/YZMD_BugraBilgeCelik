from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_metrics(original_text, summary):
    metrics = {}

    # **ROUGE-L ve BLEU Skorları**
    smoothing = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(original_text, summary)

    metrics["ROUGE-L"] = rouge_scores['rougeL'].fmeasure
    metrics["BLEU"] = sentence_bleu([original_text.split()], summary.split(), smoothing_function=smoothing)

    return metrics
