import fitz  # PyMuPDF
import re

def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def parse_portfolio_from_text(text):
    pattern = r"([A-Z]{2,5})\s*-\s*Quantity:\s*(\d+)\s*-\s*Buy Price:\s*(\d+\.?\d*)"
    matches = re.findall(pattern, text)
    portfolio = [{"symbol": m[0], "quantity": int(m[1]), "buy_price": float(m[2])} for m in matches]
    return portfolio
