from collections import defaultdict

def to_markdown(items):
    if not items:
        return "_No fresh biotech/pharma headlines in the last 24 h._"

    sections = defaultdict(list)
    for it in items:
        cat = it.get("category", "Other")
        sections[cat].append(it)
    
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
    for cat in category_order:
        if cat not in sections:
            continue
        lines.append(f"### {cat}")
        
        # Sort items: first by source (alphabetically), then by date (newest first)
        sorted_items = sorted(sections[cat], 
                            key=lambda x: (x['source'], -x['published'].timestamp()))
        
        for i in sorted_items:
            title = i["title"].strip()
            lines.append(f"- [{title}]({i['link']}) — {i['source']}")
        lines.append("")           # blank line
    return "\n".join(lines)