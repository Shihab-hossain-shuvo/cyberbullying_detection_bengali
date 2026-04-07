import re

def clean_text(text):
    """
    Basic text preprocessing (aligned with research):
    - Lowercase
    - Remove extra spaces
    """
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text