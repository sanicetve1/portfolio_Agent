
import re
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print("❌ PDF extraction error:", e)
        return ""

def parse_portfolio_from_text(text):
    try:
        pattern = re.compile(
            r"Symbol:\s*(?P<symbol>\w+),\s*Quantity:\s*(?P<quantity>\d+),\s*Buy Price:\s*(?P<buy_price>\d+(?:\.\d+)?)",
            re.IGNORECASE
        )
        matches = pattern.findall(text)
        portfolio = []
        for match in matches:
            symbol, quantity, buy_price = match
            portfolio.append({
                "symbol": symbol.strip().upper(),
                "quantity": int(quantity),
                "buy_price": float(buy_price)
            })
        print("✅ Extracted portfolio:", portfolio)
        return portfolio
    except Exception as e:
        print("❌ Portfolio parsing error:", e)
        return []
