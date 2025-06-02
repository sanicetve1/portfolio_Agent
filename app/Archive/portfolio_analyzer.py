from yahoo_fin import stock_info as si

def analyze_portfolio(portfolio):
    analysis = []
    for stock in portfolio:
        try:
            current_price = si.get_live_price(stock["symbol"])
        except:
            current_price = 0.0
        profit_loss = (current_price - stock["buy_price"]) * stock["quantity"]
        suggestion = "HOLD"
        if profit_loss > 100:
            suggestion = "SELL"
        elif profit_loss < -100:
            suggestion = "BUY MORE"
        analysis.append({
            **stock,
            "current_price": round(current_price, 2),
            "profit_loss": round(profit_loss, 2),
            "suggestion": suggestion
        })
    return analysis