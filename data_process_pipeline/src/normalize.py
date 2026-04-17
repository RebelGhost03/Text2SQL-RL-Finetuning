from typing import List
from nltk import word_tokenize, pos_tag, ne_chunk
import re
from unidecode import unidecode

from utils.text import normalize_date, normalize_quote

def extract_entities(text: str) -> List[str]:
    """
    Extract named entities from text.
    """

    entities = []

    tokens = word_tokenize(text)
    for token in tokens:
        if any(ch.isupper() for ch in token):
            entities.append(token)

    return entities

def normalize_question(text: str) -> str:
    """
    Normalize a question string by lowercasing, removing accents,
    standardizing date formats, normalizing quotation marks,
    and collapsing multiple spaces into one.
    """

    # text = text.lower()
    text = unidecode(text)
    text = normalize_date(text)
    text = normalize_quote(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_sql(sql: str) -> str:
    sql = normalize_quote(sql)
    sql = sql.replace("'", "\\'")
    sql = sql.replace('"', "'")
    return sql