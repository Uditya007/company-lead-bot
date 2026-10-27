import os
import re
import sys
import time
import warnings
import argparse
import concurrent.futures
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse, urljoin, parse_qs, unquote

# Suppress SSL/urllib3 LibreSSL warnings on macOS
warnings.filterwarnings("ignore", category=Warning)

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import db

# Try importing Gemini SDK
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class SafeConsole(Console):
    def print(self, *args, **kwargs):
        try:
            super().print(*args, **kwargs)
        except (OSError, IOError):
            pass

console = SafeConsole()

# Load environment variables
load_dotenv(override=True)
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", GOOGLE_PLACES_API_KEY).strip()

DISALLOWED_EMAIL_DOMAINS = {
    "duckduckgo.com", "google.com", "example.com", "w3.org", "sentry.io",
    "itgovernance.co.uk", "cloudflare.com", "schema.org", "wordpress.org",
    "gravatar.com", "github.com", "doubleclick.net", "facebook.com",
    "twitter.com", "instagram.com", "linkedin.com", "domain.com", "email.com",
    "sample.com", "test.com", "yoursite.com", "mysite.com", "whoisguard.com",
    "domainsbyproxy.com", "privacyguardian.org"
}


class CompanyEnrichmentBot:
    def __init__(self, google_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None, max_workers: int = 14):
        self.google_api_key = (google_api_key or GOOGLE_PLACES_API_KEY).strip()
        self.gemini_api_key = (gemini_api_key or GEMINI_API_KEY).strip()
        self.max_workers = max_workers
        self.gemini_client = None
        self.last_duplicates_skipped = 0
        
        db.init_db()

        if GENAI_AVAILABLE and self.gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception:
                self.gemini_client = None

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=30, max_retries=1)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        })

    def ai_enhance_query(self, user_query: str) -> Dict[str, str]:
        """Use AI to optimize user query into high-precision terms and targeted leadership roles."""
        clean_query = user_query.strip()
        
        roles = "CEO OR Founder OR Owner OR Director OR Managing Director OR President"
        if any(w in clean_query.lower() for w in ["dentist", "dental", "clinic", "doctor"]):
            roles = "Dentist OR Founder OR Owner OR Chief Medical Officer OR Partner"
        elif any(w in clean_query.lower() for w in ["law", "attorney", "legal"]):
            roles = "Managing Partner OR Attorney OR Partner OR Founder"
        elif any(w in clean_query.lower() for w in ["agency", "marketing", "design"]):
            roles = "Founder OR CEO OR Creative Director OR Marketing Director OR Partner"
        elif any(w in clean_query.lower() for w in ["solar", "roof", "contractor", "construction"]):
            roles = "Owner OR Founder OR President OR General Manager OR Managing Director"

        if self.gemini_client:
            try:
                prompt = (
                    f"Given this company search query: '{user_query}', return a JSON with two keys: "
                    f"'places_query': optimized clean location/business term for Google Places, and "
                    f"'target_roles': top 3 decision maker job titles separated by OR."
                )
                resp = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                if resp and resp.text:
                    return {"places_query": clean_query, "roles": roles}
            except Exception:
                pass

        return {"places_query": clean_query, "roles": roles}

    def _clean_url(self, href: str) -> str:
        """Extract direct URL from search engine redirects."""
        if not href:
            return ""
        if "duckduckgo.com/l/?" in href or "uddg=" in href:
            parsed = parse_qs(urlparse(href).query)
            target = parsed.get("uddg", [""])[0]
            if target:
                return unquote(target)
        return href

    def search_web(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Fast web search with clean URL resolution."""
        results = []
        try:
            url = "https://lite.duckduckgo.com/lite/"
            resp = self.session.post(url, data={"q": query}, timeout=4)
            soup = BeautifulSoup(resp.text, "html.parser")
            
            links = soup.select("a.result-link")
            snippets = soup.select("td.result-snippet")
            
            for i, link in enumerate(links[:limit * 3]):
                title = link.get_text(strip=True)
                raw_href = link.get("href", "")
                href = self._clean_url(raw_href)
                snippet = snippets[i].get_text(strip=True) if i < len(snippets) else ""
                
                if href and title and "duckduckgo.com" not in href:
                    results.append({
                        "title": title,
                        "href": href,
                        "snippet": snippet
                    })
                if len(results) >= limit:
                    break
        except Exception:
            pass
        return results

    def search_companies(
        self,
        query: str,
        limit: int = 10,
        only_new_listed: bool = False,
        only_no_website: bool = False,
        exclude_already_enriched: bool = True
    ) -> List[Dict]:
        """
        Search for companies with strict verification:
        - Exclude previously enriched leads (if enabled)
        - Filter for newly listed / emerging establishments (if enabled)
        - Filter for businesses without websites (if enabled)
        - Skip closed businesses
        """
        existing_ids, existing_domains, existing_names = (set(), set(), set())
        if exclude_already_enriched:
            existing_ids, existing_domains, existing_names = db.get_already_enriched_identifiers()
            console.print(f"[cyan]➜ [Lead Memory] Active. Checking against {len(existing_ids)} previously saved leads...[/cyan]")

        if self.google_api_key and self.google_api_key != "your_google_places_api_key_here":
            console.print(f"[cyan]➜ [Google Places API (New)] Fetching real verified listings (No Website Only: {only_no_website})...[/cyan]")
            places_results = self._search_google_places_new(
                query=query,
                limit=limit,
                only_new_listed=only_new_listed,
                only_no_website=only_no_website,
                existing_ids=existing_ids,
                existing_domains=existing_domains,
                existing_names=existing_names
            )
            if places_results:
                return places_results
            console.print("[yellow]➜ Places API returned 0 matching results. Falling back to web search...[/yellow]")
            return self._search_free_web(query, limit=limit, existing_domains=existing_domains, existing_names=existing_names)
        else:
            console.print(f"[yellow]➜ Using AI-enhanced web search fallback...[/yellow]")
            return self._search_free_web(query, limit=limit, existing_domains=existing_domains, existing_names=existing_names)

    def generate_search_queries(
        self,
        query: str,
        limit: int,
        only_new_listed: bool = False,
        only_no_website: bool = False
    ) -> List[str]:
        """
        Generate multiple targeted search queries across geographic sub-regions,
        directions, and business synonyms to guarantee collecting up to 50-100 leads
        even after strict deduplication and filters.
        """
        queries = [query]
        clean_q = query.strip()
        lower_q = clean_q.lower()

        # If query has 'in' or 'near' (e.g. 'dentists in Bengaluru' or 'plumbers in Miami')
        location = ""
        category = ""
        for sep in [" in ", " near ", " around ", " at "]:
            if sep in lower_q:
                idx = lower_q.find(sep)
                category = clean_q[:idx].strip()
                location = clean_q[idx + len(sep):].strip()
                break

        if location and category:
            # Sub-region / directional expansion
            queries.append(f"top {category} in {location}")
            queries.append(f"best {category} in {location}")
            queries.append(f"{category} in South {location}")
            queries.append(f"{category} in North {location}")
            queries.append(f"{category} in East {location}")
            queries.append(f"{category} in West {location}")
            queries.append(f"{category} in Central {location}")
            queries.append(f"{category} in Downtown {location}")
            queries.append(f"local {category} in {location}")
            
            # Popular city hubs if Bengaluru / Bangalore
            if "bengaluru" in location.lower() or "bangalore" in location.lower():
                queries.append(f"{category} in Koramangala Bengaluru")
                queries.append(f"{category} in Indiranagar Bengaluru")
                queries.append(f"{category} in Whitefield Bengaluru")
                queries.append(f"{category} in HSR Layout Bengaluru")
                queries.append(f"{category} in Jayanagar Bengaluru")
                queries.append(f"{category} in Electronic City Bengaluru")
        else:
            queries.append(f"best {clean_q}")
            queries.append(f"top {clean_q}")
            queries.append(f"local {clean_q}")
            queries.append(f"{clean_q} services")
            queries.append(f"independent {clean_q}")

        if only_new_listed:
            new_variants = [f"newly opened {q}" for q in queries[:5]] + [f"new {q}" for q in queries[:5]]
            queries = new_variants + queries
        elif only_no_website:
            no_web_variants = [f"local {q}" for q in queries[:5]] + [f"independent {q}" for q in queries[:5]]
            queries = no_web_variants + queries

        # Deduplicate while preserving order
        unique_queries = []
        seen = set()
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        return unique_queries

    def _search_google_places_new(
        self,
        query: str,
        limit: int = 10,
        only_new_listed: bool = False,
        only_no_website: bool = False,
        existing_ids: Optional[Set[str]] = None,
        existing_domains: Optional[Set[str]] = None,
        existing_names: Optional[Set[str]] = None
    ) -> List[Dict]:
        """Query Google Places API (New) with exact field masks, verification, deduplication, newly-listed, and no-website filtering."""
        existing_ids = existing_ids or set()
        existing_domains = existing_domains or set()
        existing_names = existing_names or set()
        self.last_duplicates_skipped = 0

        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.google_api_key,
            "X-Goog-FieldMask": (
                "places.id,places.displayName,places.formattedAddress,"
                "places.nationalPhoneNumber,places.internationalPhoneNumber,"
                "places.websiteUri,places.rating,places.userRatingCount,"
                "places.businessStatus,places.googleMapsUri,places.primaryTypeDisplayName,"
                "places.editorialSummary,nextPageToken"
            )
        }

        # Dynamically generate multi-query variations to guarantee reaching 50-100 leads
        search_queries = self.generate_search_queries(
            query=query,
            limit=limit,
            only_new_listed=only_new_listed,
            only_no_website=only_no_website
        )

        candidate_places = []
        seen_place_ids = set()
        target_count = limit if not only_new_listed else limit * 2

        for q in search_queries:
            if len(candidate_places) >= target_count:
                break

            next_page_token = None
            max_pages_per_query = 3
            page_count = 0

            while page_count < max_pages_per_query:
                page_count += 1
                data = {
                    "textQuery": q,
                    "pageSize": 20
                }
                if next_page_token:
                    data["pageToken"] = next_page_token

                try:
                    resp = self.session.post(url, headers=headers, json=data, timeout=8)
                    if resp.status_code == 200:
                        resp_json = resp.json()
                        raw_places = resp_json.get("places", [])
                        if not raw_places:
                            break

                        for p in raw_places:
                            status = p.get("businessStatus", "OPERATIONAL")
                            if status in ["CLOSED_PERMANENTLY", "CLOSED_TEMPORARILY"]:
                                continue

                            place_id = p.get("id", "")
                            if not place_id or place_id in seen_place_ids:
                                continue

                            name_obj = p.get("displayName", {})
                            name = name_obj.get("text", "") if isinstance(name_obj, dict) else str(name_obj)
                            website = p.get("websiteUri", "")
                            has_web = bool(website and website.strip())

                            # If user requested only businesses without website, skip those that have one
                            if only_no_website and has_web:
                                continue

                            # Check lead memory database
                            if db.is_lead_duplicate(place_id, website, name, existing_ids, existing_domains, existing_names):
                                self.last_duplicates_skipped += 1
                                continue

                            seen_place_ids.add(place_id)

                            user_ratings = p.get("userRatingCount")
                            user_ratings_count = int(user_ratings) if user_ratings is not None else 0

                            phone = p.get("nationalPhoneNumber") or p.get("internationalPhoneNumber", "")
                            type_obj = p.get("primaryTypeDisplayName", {})
                            category = type_obj.get("text", "") if isinstance(type_obj, dict) else ""
                            summary_obj = p.get("editorialSummary", {})
                            summary = summary_obj.get("text", "") if isinstance(summary_obj, dict) else ""

                            candidate_places.append({
                                "place_id": place_id,
                                "name": name,
                                "address": p.get("formattedAddress", ""),
                                "phone": phone,
                                "website": website,
                                "has_website": has_web,
                                "has_no_website": not has_web,
                                "rating": str(p.get("rating", "")),
                                "user_rating_count": user_ratings_count,
                                "category": category,
                                "summary": summary,
                                "google_maps_uri": p.get("googleMapsUri", ""),
                                "is_google_verified": True
                            })

                            if len(candidate_places) >= target_count:
                                break

                        next_page_token = resp_json.get("nextPageToken")
                        if not next_page_token or len(candidate_places) >= target_count:
                            break
                        time.sleep(0.3)
                    else:
                        break
                except Exception:
                    break

            if only_new_listed and len(candidate_places) >= limit * 2:
                break

        if not candidate_places:
            return []

        # If "Only Newly Listed" is requested, sort candidates by lowest review count (ascending)
        if only_new_listed:
            candidate_places.sort(key=lambda x: x.get("user_rating_count", 0))

        # Mark "is_new_listing" appropriately
        final_places = candidate_places[:limit]
        for p in final_places:
            rev_cnt = p.get("user_rating_count", 0)
            p["is_new_listing"] = (rev_cnt <= 25) or only_new_listed
            # Add to deduplication tracking
            if p.get("place_id"):
                existing_ids.add(p["place_id"])
            if p.get("website"):
                existing_domains.add(db.clean_domain_name(p["website"]))

        return final_places

    def _search_free_web(
        self,
        query: str,
        limit: int = 10,
        existing_domains: Optional[Set[str]] = None,
        existing_names: Optional[Set[str]] = None
    ) -> List[Dict]:
        existing_domains = existing_domains or set()
        existing_names = existing_names or set()

        results = []
        raw_results = self.search_web(query, limit=limit * 4)

        ignored_domains = [
            "yelp.", "tripadvisor.", "yellowpages.", "wikipedia.org", "facebook.com", 
            "instagram.com", "f6s.com", "clutch.co", "builtin.", "inven.ai", 
            "reddit.com", "linkedin.com", "zoominfo.com", "g2.com", "glassdoor.com",
            "bloomberg.com", "crunchbase.com", "dnb.com", "indeed.com"
        ]

        for item in raw_results:
            title = item.get("title", "")
            href = item.get("href", "")
            snippet = item.get("snippet", "")

            domain = urlparse(href).netloc.lower()
            if any(ign in domain for ign in ignored_domains):
                continue

            clean_name = title.split(" - ")[0].split(" | ")[0].split(" : ")[0].strip()
            
            # Check duplicate
            clean_dom = db.clean_domain_name(href)
            if clean_dom in existing_domains:
                continue

            clean_n = re.sub(r'[^a-z0-9]', '', clean_name.lower())
            if clean_n and clean_n in existing_names:
                continue

            results.append({
                "place_id": f"web_{clean_dom or clean_n}",
                "name": clean_name,
                "address": "Web Result",
                "phone": "",
                "website": href,
                "rating": "",
                "user_rating_count": 0,
                "is_new_listing": False,
                "category": "Business / Organization",
                "summary": snippet[:150],
                "google_maps_uri": "",
                "is_google_verified": False
            })

            existing_domains.add(clean_dom)

            if len(results) >= limit:
                break

        return results

    def _validate_phone_number(self, phone_str: str) -> str:
        """Validate that a scraped phone number is a genuine phone number and not random digits/timestamp."""
        if not phone_str:
            return ""
        clean = phone_str.strip()
        # Discard obvious bad matches like timestamps (e.g. 1775236631) or decimal versions
        if "." in clean or "/" in clean or "@" in clean or len(clean) > 22:
            return ""
        digits = re.sub(r'\D', '', clean)
        # Genuine phone numbers have 10 to 14 digits
        if not (10 <= len(digits) <= 14):
            return ""
        # Discard repeated fake patterns like 0000000000, 1111111111, 1234567890
        if len(set(digits)) <= 2 or digits == "1234567890":
            return ""
        # Format if 10-digit US/CA number
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits.startswith("1"):
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        return clean

    def _validate_email(self, email_str: str, website_url: str) -> bool:
        """Strictly validate that an email address is real, has a valid TLD, and is not a tracking/dummy email."""
        if not email_str or "@" not in email_str:
            return False
        clean = email_str.strip().lower()
        parts = clean.split("@")
        if len(parts) != 2:
            return False
        user_part, domain_part = parts
        
        # Discard dummy domains
        if domain_part in DISALLOWED_EMAIL_DOMAINS:
            return False
        if any(dummy in domain_part for dummy in ["sentry", "w3.org", "cloudflare", "domain.com", "example"]):
            return False

        # Discard file extensions masquerading as emails
        ext = domain_part.split(".")[-1]
        if ext in ["png", "jpg", "jpeg", "gif", "svg", "webp", "css", "js", "html", "php", "woff", "pdf", "ico"]:
            return False

        # Valid top-level domain
        if len(ext) < 2 or not ext.isalpha():
            return False

        # Reject user parts that are obviously system strings
        if user_part in ["username", "email", "name", "yourname", "test", "user", "author"]:
            return False

        return True

    def scrape_and_analyze_company(self, website_url: str, company_name: str, base_summary: str = "", existing_phone: str = "") -> Dict[str, str]:
        """Extract genuine emails, validate phones, and generate an AI business profile."""
        emails = set()
        phones = set()
        extracted_text_chunks = []
        
        # If we already have a Google Maps verified phone number, keep it as verified primary
        if existing_phone:
            phones.add(existing_phone.strip())

        if website_url and website_url.startswith("http") and "duckduckgo.com" not in website_url:
            urls_to_check = [website_url]
            try:
                parsed = urlparse(website_url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                urls_to_check.append(urljoin(base_url, "/contact"))
                urls_to_check.append(urljoin(base_url, "/about"))
            except Exception:
                pass

            email_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
            phone_pattern = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')

            for target_url in urls_to_check[:2]:
                try:
                    resp = self.session.get(target_url, timeout=2.8)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        
                        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
                        if meta_desc and meta_desc.get('content'):
                            extracted_text_chunks.append(meta_desc['content'].strip())

                        # 1. Mailto links (highest confidence)
                        for a_tag in soup.select('a[href^="mailto:"]'):
                            raw_email = a_tag['href'].replace('mailto:', '').split('?')[0].strip()
                            if self._validate_email(raw_email, website_url):
                                emails.add(raw_email.lower())

                        # 2. Tel links
                        if not phones:
                            for a_tag in soup.select('a[href^="tel:"]'):
                                raw_tel = a_tag['href'].replace('tel:', '').strip()
                                valid_phone = self._validate_phone_number(raw_tel)
                                if valid_phone:
                                    phones.add(valid_phone)

                        # 3. Regex scan text for emails
                        for e in email_pattern.findall(resp.text):
                            if self._validate_email(e, website_url):
                                emails.add(e.lower())

                        # 4. Strict phone regex only if no phone exists yet
                        if not phones:
                            for p in phone_pattern.findall(resp.text):
                                valid_phone = self._validate_phone_number(p)
                                if valid_phone:
                                    phones.add(valid_phone)
                except Exception:
                    continue

        # Sort emails so that domain-matching company emails appear first!
        clean_site_domain = db.clean_domain_name(website_url)
        sorted_emails = sorted(
            list(emails),
            key=lambda e: 0 if (clean_site_domain and clean_site_domain in e.split("@")[-1]) else 1
        )

        combined_text = " ".join(extracted_text_chunks) or base_summary
        ai_summary = ""
        
        if self.gemini_client and combined_text:
            try:
                prompt = f"In 1 crisp sentence, describe what {company_name} offers based on: {combined_text[:400]}"
                resp = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                if resp and resp.text:
                    ai_summary = resp.text.strip()
            except Exception:
                pass

        if not ai_summary:
            if not website_url:
                ai_summary = f"Operating as {base_summary or company_name} without an official website listed on Google Maps. High-intent prospect for web design, local SEO, and digital booking services."
            elif combined_text:
                clean_snippet = re.sub(r'\s+', ' ', combined_text).strip()
                ai_summary = clean_snippet[:160] + "..." if len(clean_snippet) > 160 else clean_snippet
            else:
                ai_summary = f"Commercial business specializing in {company_name} products & services."

        return {
            "email": ", ".join(sorted_emails[:2]),
            "phone": list(phones)[0] if phones else "",
            "ai_summary": ai_summary
        }

    def find_linkedin_and_employees(self, company_name: str, role_keywords: str = "CEO OR Founder OR Director OR Owner") -> Dict[str, str]:
        """Search and strictly verify real LinkedIn company profiles and executive personnel."""
        linkedin_company = ""
        employees = []

        # 1. Company LinkedIn (Must be valid company profile)
        company_results = self.search_web(f'site:linkedin.com/company "{company_name}"', limit=2)
        for r in company_results:
            href = r.get("href", "")
            # Verify exact company slug structure: linkedin.com/company/xxx
            if re.search(r'linkedin\.com/company/[a-zA-Z0-9_\-]+', href):
                if not any(bad in href for bad in ["/jobs/", "/pulse/", "/posts/", "/people/"]):
                    linkedin_company = href.split("?")[0].rstrip("/")
                    break

        # 2. Verified Decision Makers
        people_results = self.search_web(f'site:linkedin.com/in "{company_name}" ({role_keywords})', limit=4)
        for p in people_results:
            title = p.get("title", "")
            link = p.get("href", "")
            
            # Verify personal LinkedIn profile structure: linkedin.com/in/xxx
            if re.search(r'linkedin\.com/in/[a-zA-Z0-9_\-]+', link):
                clean_title = title.split(" - ")[0].split(" | ")[0].replace(" | LinkedIn", "").strip()
                # Check for executive keywords to ensure real decision-maker match
                if any(role.lower() in clean_title.lower() for role in ["ceo", "founder", "owner", "president", "director", "partner", "chief", "dentist", "managing"]):
                    employees.append(f"{clean_title} ({link.split('?')[0]})")
            
            if len(employees) >= 3:
                break

        return {
            "linkedin_company": linkedin_company,
            "employees": " \n".join(employees) if employees else "None identified"
        }

    def enrich_single_company(self, comp: Dict, roles: str) -> Dict:
        """Worker function to enrich a single company concurrently."""
        name = comp.get("name", "Unknown")
        website = comp.get("website", "")
        summary = comp.get("summary", "")
        existing_phone = comp.get("phone", "")

        # 1. Contact analysis & AI summary
        analysis = self.scrape_and_analyze_company(
            website_url=website,
            company_name=name,
            base_summary=summary,
            existing_phone=existing_phone
        )
        comp["email"] = analysis.get("email", "")
        if not comp.get("phone"):
            comp["phone"] = analysis.get("phone", "")
        comp["ai_summary"] = analysis.get("ai_summary", "")

        # 2. LinkedIn & Verified Decision Makers
        social_info = self.find_linkedin_and_employees(name, role_keywords=roles)
        comp["linkedin_company"] = social_info.get("linkedin_company", "")
        comp["employees"] = social_info.get("employees", "")

        return comp

    def run(
        self,
        query: str,
        limit: int = 10,
        only_new_listed: bool = False,
        only_no_website: bool = False,
        exclude_already_enriched: bool = True,
        output_format: str = "csv",
        output_path: Optional[str] = None
    ):
        console.print(f"\n[bold green]⚡ [Parallel AI Engine] Searching:[/bold green] [bold white]'{query}'[/bold white] (Limit: {limit}, Newly Listed: {only_new_listed}, No Website: {only_no_website}, Deduplication: {exclude_already_enriched})\n")

        ai_query_data = self.ai_enhance_query(query)
        search_term = ai_query_data.get("places_query", query)
        roles = ai_query_data.get("roles", "CEO OR Founder OR Director")

        # 1. Fetch places with deduplication, newly listed, and no-website filtering
        companies = self.search_companies(
            query=search_term,
            limit=limit,
            only_new_listed=only_new_listed,
            only_no_website=only_no_website,
            exclude_already_enriched=exclude_already_enriched
        )
        if not companies:
            console.print("[bold red]No new companies found matching these criteria. Try adjusting your query or toggle.[/bold red]")
            return None

        console.print(f"[bold green]✔ Found {len(companies)} matching businesses.[/bold green] Running parallel enrichment...\n")

        # 2. Multi-threaded parallel enrichment
        enriched_data = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Enriching {len(companies)} leads...", total=len(companies))

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_comp = {
                    executor.submit(self.enrich_single_company, comp, roles): comp for comp in companies
                }
                for future in concurrent.futures.as_completed(future_to_comp):
                    try:
                        enriched_item = future.result()
                        enriched_data.append(enriched_item)
                    except Exception:
                        pass
                    progress.advance(task)

        # 3. Save into memory database for future deduplication!
        db.save_leads_batch(enriched_data, query_tag=query)
        console.print(f"[green]✔ Automatically stored {len(enriched_data)} leads into persistent lead memory.[/green]")

        df = pd.DataFrame(enriched_data)

        clean_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', query.lower())[:30]
        if not output_path:
            output_path = f"leads_{clean_filename}.{output_format}"

        if output_format == "xlsx":
            df.to_excel(output_path, index=False)
        elif output_format == "json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            df.to_csv(output_path, index=False)

        self.display_summary_table(enriched_data[:10])
        console.print(f"\n[bold green]✔ Successfully saved {len(enriched_data)} leads to:[/bold green] [bold underline yellow]{os.path.abspath(output_path)}[/bold underline yellow]\n")
        return df

    def display_summary_table(self, data: List[Dict]):
        table = Table(title="AI-Enriched Company Leads Preview", show_header=True, header_style="bold magenta")
        table.add_column("Company Name", style="bold cyan", max_width=20)
        table.add_column("Phone", style="green", max_width=16)
        table.add_column("Email", style="yellow", max_width=22)
        table.add_column("New?", style="bold green", max_width=8)
        table.add_column("AI Summary", style="white", max_width=30)
        table.add_column("LinkedIn", style="magenta", max_width=22)

        for row in data:
            table.add_row(
                str(row.get("name", "")),
                str(row.get("phone", "")),
                str(row.get("email", "")),
                "🌱 Yes" if row.get("is_new_listing") else "No",
                str(row.get("ai_summary", "")),
                str(row.get("linkedin_company", ""))
            )

        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Verified AI Google Maps & Company Lead Enrichment Bot")
    parser.add_argument("-q", "--query", type=str, help="Search query (e.g. 'AI Startups in Austin TX')")
    parser.add_argument("-l", "--limit", type=int, default=10, help="Maximum number of companies to extract (default: 10)")
    parser.add_argument("--new-only", action="store_true", help="Show only newly listed / emerging businesses (<= 10 reviews)")
    parser.add_argument("--no-website-only", action="store_true", help="Show only businesses without a website")
    parser.add_argument("--include-duplicates", action="store_true", help="Include previously enriched leads (disabled by default)")
    parser.add_argument("-f", "--format", choices=["csv", "xlsx", "json"], default="csv", help="Export format (default: csv)")
    parser.add_argument("-o", "--output", type=str, help="Custom output file path")
    parser.add_argument("-w", "--workers", type=int, default=14, help="Number of concurrent worker threads")

    args = parser.parse_args()

    query = args.query
    if not query:
        console.print("\n[bold cyan]=== Verified AI Lead Enrichment Bot ===[/bold cyan]")
        query = console.input("[bold yellow]Enter your search query (e.g. 'AI startups in Boston'): [/bold yellow]").strip()
        if not query:
            query = "AI startups in Boston"

    bot = CompanyEnrichmentBot(max_workers=args.workers)
    bot.run(
        query=query,
        limit=args.limit,
        only_new_listed=args.new_only,
        only_no_website=args.no_website_only,
        exclude_already_enriched=(not args.include_duplicates),
        output_format=args.format,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
