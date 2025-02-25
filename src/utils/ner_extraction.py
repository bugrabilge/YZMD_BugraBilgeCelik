import spacy
from dateutil.parser import parse
import re

nlp = spacy.load("en_core_web_trf")


def is_valid_date(text):
    """Metin gerçekten bir tarih mi kontrol eder."""
    try:
        parse(text, fuzzy=True)
        return True
    except ValueError:
        return False


def normalize_date(text):
    """Tarih formatlarını normalize eder."""
    try:
        return parse(text, fuzzy=True).strftime("%Y-%m-%d")
    except ValueError:
        return text


def extract_dates_locations(text):
    """Verilen metinden tarih ve lokasyon bilgilerini çıkarır."""
    doc = nlp(text)

    # Daha doğru tarih çıkarımı için ek filtreleme ve normalizasyon
    dates = [normalize_date(ent.text) for ent in doc.ents if ent.label_ == "DATE" and is_valid_date(ent.text)]

    # Lokasyonları çeşitlendirme (ülke, şehir, bölge, coğrafi alanları ekleme)
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC", "NORP"]]

    # Tarih ve lokasyonları tekrarsız hale getirme
    dates = sorted(list(set(dates)))
    locations = sorted(list(set(locations)))

    return {"dates": dates, "locations": locations}
