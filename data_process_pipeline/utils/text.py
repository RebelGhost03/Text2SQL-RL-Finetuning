import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime

STOPWORDS = set(stopwords.words("english"))

def text_to_canonical_form(text: str) -> str:
    """
    Normalize text into a canonical form:
    - Split CamelCase / PascalCase / ALLCAPS into words
    - Lowercase
    - Remove stopwords
    - Apply stemming
    """

    stemmer = PorterStemmer()
    
    # ThisText -> This Text
    # thisText -> this Text
    # THISText -> THIS Text
    text = " ".join(re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|\d+|[A-Z]+", text))

    text = text.lower().replace('_', ' ')
    words = [stemmer.stem(w) for w in text.split() if w not in STOPWORDS]

    return " ".join(words)

def normalize_date(text: str, out_format: str = "%Y-%m-%d") -> str:
    """
    Find date strings in text, parse them using known formats, and normalize them into a single output date format.
    """

    DATE_REGEX_FORMATS = [
        # YYYY-MM-DD or YYYY/MM/DD
        (r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "%Y-%m-%d"),

        # DD-MM-YYYY or DD/MM/YYYY
        (r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", "%d-%m-%Y"),
    ]

    for pattern, in_fmt in DATE_REGEX_FORMATS:
        for date_str in re.findall(pattern, text):
            try:
                dt = datetime.strptime(
                    date_str.replace("/", "-"),
                    in_fmt
                )
                text = text.replace(date_str, dt.strftime(out_format))
            except ValueError:
                pass
    
    return text

def normalize_quote(text: str) -> str:
    """
    Convert single quotes used as quotation marks into double quotes (""), 
    while preserving apostrophes inside words (e.g. "John's", "John''s").
    """

    single_quote_ids = []
    is_start_quote = True
    for i, ch in enumerate(text):
        if ch == "'":
            if is_start_quote:
                if i > 0 and (text[i - 1].isalpha() or text[i - 1] == "'"):
                    # If this quote is open quote and previous is alpha (doesn't, I'm, ours', ...) or single quote (John''s)
                    single_quote_ids.append(i)
                else:
                    is_start_quote = not is_start_quote
            else:
                is_start_quote = not is_start_quote
    for i in single_quote_ids:
        text = text[:i] + "@$" + text[i + 1:]
    
    text = text.replace("'", '"')
    text = re.sub(r'(@\$)+', "'", text)

    return text

def to_snake_case(text: str) -> str:
    text = re.sub(r"[ \-]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = text.lower()
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text

def fix_unicode(text: str) -> str:
    if not isinstance(text, str):
        return text

    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except:
        pass

    try:
        text = text.encode("latin1").decode("utf-8")
    except:
        pass

    return text