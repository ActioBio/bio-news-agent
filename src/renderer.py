from collections import defaultdict

def to_markdown(items):
    if not items:
        return "_No fresh biotech/pharma headlines in the last 24 h._"

    sections = defaultdict(list)
    for it in items:
        cat = it.get("category", "Other")
        sections[cat].append(it)

    # Debug: print what categories we actually have
    print(f"📊 Categories found: {list(sections.keys())}")
    
    lines = ["## Daily Biotech / Pharma Headlines\n"]
    
    # Define category order
    category_order = [
        "Regulatory & FDA",
        "Clinical & Research", 
        "Deals & Finance",
        "Company News",
        "Policy & Politics",
        "Market Insights",
        "Other"
    ]
    
    # Add any categories not in our predefined order
    for cat in sections:
        if cat not in category_order:
            category_order.append(cat)
    
    for cat in category_order:
        if cat not in sections:
            continue
        lines.append(f"### {cat}")
        for i in sections[cat]:
            title = i["title"].strip()
            if "blurb" in i:       # added later by the LLM agent
                title += f" — *{i['blurb']}*"
            lines.append(f"- [{title}]({i['link']}) — {i['source']}")
        lines.append("")           # blank line
    return "\n".join(lines)