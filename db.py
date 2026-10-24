import sqlite3
import os
import re
from typing import Set, Tuple, List, Dict, Optional
from urllib.parse import urlparse

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leads_memory.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enriched_leads (
            place_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            clean_domain TEXT,
            address TEXT,
            phone TEXT,
            email TEXT,
            website TEXT,
            rating TEXT,
            user_rating_count INTEGER DEFAULT 0,
            category TEXT,
            summary TEXT,
            ai_summary TEXT,
            linkedin_company TEXT,
            employees TEXT,
            google_maps_uri TEXT,
            is_new_listing INTEGER DEFAULT 0,
            query_tag TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_clean_domain ON enriched_leads(clean_domain)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON enriched_leads(name)")
    conn.commit()
    conn.close()

def clean_domain_name(url: str) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def get_already_enriched_identifiers() -> Tuple[Set[str], Set[str], Set[str]]:
    """Returns sets of (place_ids, clean_domains, normalized_names) for instant deduplication."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT place_id, clean_domain, name FROM enriched_leads")
    rows = cursor.fetchall()
    conn.close()

    place_ids = set()
    domains = set()
    names = set()

    for r in rows:
        if r["place_id"]:
            place_ids.add(r["place_id"].strip())
        if r["clean_domain"]:
            domains.add(r["clean_domain"].strip().lower())
        if r["name"]:
            # Normalize name: lowercase, strip non-alphanumeric
            clean_name = re.sub(r'[^a-z0-9]', '', r["name"].lower())
            if clean_name:
                names.add(clean_name)

    return place_ids, domains, names

def is_lead_duplicate(place_id: str, website: str, name: str, existing_ids: Set[str], existing_domains: Set[str], existing_names: Set[str]) -> bool:
    """Checks if a lead has already been enriched in previous searches."""
    if place_id and place_id.strip() in existing_ids:
        return True
    
    domain = clean_domain_name(website)
    if domain and domain in existing_domains:
        return True

    clean_n = re.sub(r'[^a-z0-9]', '', (name or "").lower())
    if clean_n and len(clean_n) > 3 and clean_n in existing_names:
        return True

    return False

def save_leads_batch(leads: List[Dict], query_tag: str = "") -> int:
    """Saves newly enriched leads into memory database."""
    if not leads:
        return 0
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    saved_count = 0

    for l in leads:
        place_id = l.get("place_id") or l.get("id") or ""
        if not place_id:
            # Generate deterministic fallback ID from domain or name
            clean_d = clean_domain_name(l.get("website", ""))
            clean_n = re.sub(r'[^a-z0-9]', '', (l.get("name", "")).lower())
            place_id = f"custom_{clean_d or clean_n}"

        clean_d = clean_domain_name(l.get("website", ""))
        is_new = 1 if l.get("is_new_listing") else 0
        user_ratings = int(l.get("user_rating_count") or 0)

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO enriched_leads (
                    place_id, name, clean_domain, address, phone, email, website,
                    rating, user_rating_count, category, summary, ai_summary,
                    linkedin_company, employees, google_maps_uri, is_new_listing, query_tag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                place_id,
                l.get("name", ""),
                clean_d,
                l.get("address", ""),
                l.get("phone", ""),
                l.get("email", ""),
                l.get("website", ""),
                str(l.get("rating", "")),
                user_ratings,
                l.get("category", ""),
                l.get("summary", ""),
                l.get("ai_summary", ""),
                l.get("linkedin_company", ""),
                l.get("employees", ""),
                l.get("google_maps_uri", ""),
                is_new,
                query_tag
            ))
            saved_count += 1
        except Exception:
            continue

    conn.commit()
    conn.close()
    return saved_count

def get_total_enriched_count() -> int:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM enriched_leads")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def clear_all_history() -> bool:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM enriched_leads")
    conn.commit()
    conn.close()
    return True
