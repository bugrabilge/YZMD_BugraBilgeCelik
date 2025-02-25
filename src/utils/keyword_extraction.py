import yake

def extract_keywords(text, num_keywords=3):
    """Verilen metinden en önemli anahtar kelimeleri çıkarır."""
    kw_extractor = yake.KeywordExtractor(n=1, top=num_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]
