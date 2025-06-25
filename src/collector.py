import feedparser, hashlib, json, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_DAY = timedelta(days=1)

# Location of feeds configuration file (project root)
_FEEDS_FILE = Path(__file__).resolve().parent.parent / "feeds.json"

try:
    FEEDS = json.loads(_FEEDS_FILE.read_text())
    print(f"📰 Loaded {len(FEEDS)} feeds from feeds.json")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"❌ Error loading feeds.json: {e}")
    FEEDS = {}

def _now():
    return datetime.now(timezone.utc)

def _parse_date(entry):
    for attr in ("published_parsed", "updated_parsed"):
        tup = entry.get(attr)
        if tup:
            return datetime.fromtimestamp(time.mktime(tup), tz=timezone.utc)
    return None

def collect_items():
    """Return list[dict] fresh within 24 h."""
    cutoff = _now() - _DAY
    print(f"🕐 Collecting items newer than {cutoff}")
    items = []
    
    for url, meta in FEEDS.items():
        category = meta["category"]
        src = meta["source"]
        try:
            print(f"📡 Fetching {src}...")
            parsed = feedparser.parse(url)
            
            if parsed.bozo:
                print(f"⚠️  Parse error for {src}: {parsed.bozo_exception}")
                continue
                
            entries_count = len(parsed.entries) if hasattr(parsed, 'entries') else 0
            print(f"   Found {entries_count} entries")
            
        except Exception as exc:
            print(f"⚠️  Feed error for {url}: {exc}")
            continue
            
        for e in parsed.entries:
            ts = _parse_date(e)
            if not ts:
                continue
                
            if ts < cutoff:
                continue
                
            title, link = e.get("title", "").strip(), e.get("link", "").strip()
            if not (title and link):
                continue
                
            uid = hashlib.sha1(link.encode()).hexdigest()
            items.append(
                {
                    "id": uid,
                    "title": title,
                    "link": link,
                    "source": src,
                    "published": ts,
                    "category": category,
                }
            )
            
    print(f"✅ Collected {len(items)} total items from all feeds")
    return items