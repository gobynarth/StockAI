def format_entry_with_date(price, entry_date):
    price_html = f"${float(price):,.2f}"
    if entry_date in (None, ""):
        return price_html
    return f'{price_html}<br><span class="muted">Entered {entry_date}</span>'
