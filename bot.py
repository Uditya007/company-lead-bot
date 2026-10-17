import os
import re
import sys
import time
import warnings
import argparse
import concurrent.futures
from typing import List, Dict, Optional
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

# Try importing Gemini SDK
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

console = Console()

# Load environment variables
load_dotenv(override=True)
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", GOOGLE_PLACES_API_KEY).strip()


class CompanyEnrichmentBot:
    def __init__(self, google_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None, max_workers: int = 12):
        self.google_api_key = (google_api_key or GOOGLE_PLACES_API_KEY).strip()
        self.gemini_api_key = (gemini_api_key or GEMINI_API_KEY).strip()
        self.max_workers = max_workers
        self.gemini_client = None
        
        if GENAI_AVAILABLE and self.gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception:
                self.gemini_client = None

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=1)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        })

    def ai_enhance_query(self, user_query: str) -> Dict[str, str]:
        """Use AI to optimize the user search query into precise search terms and targeted employee roles."""
        clean_query = user_query.strip()
        
        roles = "CEO OR Founder OR Owner OR Director OR Managing Director OR VP"
        if any(w in clean_query.lower() for w in ["dentist", "dental", "clinic", "doctor"]):
            roles = "Dentist OR Founder OR Owner OR Chief Medical Officer OR Partner"
        elif any(w in clean_query.lower() for w in ["law", "attorney", "legal"]):
            roles = "Managing Partner OR Attorney OR Partner OR Founder"
        elif any(w in clean_query.lower() for w in ["agency", "marketing", "design"]):
            roles = "Founder OR CEO OR Creative Director OR Marketing Director OR Partner"

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
        """Extract destination URL if wrapped in a search engine redirect."""
        if not href:
            return ""
        if "duckduckgo.com/l/?" in href or "uddg=" in href:
            parsed = parse_qs(urlparse(href).query)
            target = parsed.get("uddg", [""])[0]
            if target:
                return unquote(target)
        return href

    def search_web(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Perform fast search using DuckDuckGo lite interface."""
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

    def search_companies(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for companies using Google Places API (New) or free web search."""
        if self.google_api_key and self.google_api_key != "your_google_places_api_key_here":
            console.print(f"[cyan]➜ [AI + Maps] Querying Google Places API (New) for up to {limit} places...[/cyan]")
            places_results = self._search_google_places_new(query, limit=limit)
            if places_results:
                return places_results
            console.print("[yellow]➜ Places API returned 0 results. Falling back to web search...[/yellow]")
            return self._search_free_web(query, limit=limit)
        else:
            console.print(f"[yellow]➜ Using AI-enhanced web search fallback for up to {limit} results...[/yellow]")
            return self._search_free_web(query, limit=limit)

    def _search_google_places_new(self, query: str, limit: int = 5) -> List[Dict]:
        """Call modern Google Places API (New) endpoint with fast multi-page pagination."""
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.google_api_key,
            "X-Goog-FieldMask": (
                "places.displayName,places.formattedAddress,"
                "places.nationalPhoneNumber,places.internationalPhoneNumber,"
                "places.websiteUri,places.rating,places.primaryTypeDisplayName,places.editorialSummary,nextPageToken"
            )
        }
        
        all_places = []
        next_page_token = None
        
        while len(all_places) < limit:
            page_size = min(limit - len(all_places), 20)
            data = {
                "textQuery": query,
                "pageSize": page_size
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
                        name_obj = p.get("displayName", {})
                        name = name_obj.get("text", "") if isinstance(name_obj, dict) else str(name_obj)
                        phone = p.get("nationalPhoneNumber") or p.get("internationalPhoneNumber", "")
                        
                        type_obj = p.get("primaryTypeDisplayName", {})
                        category = type_obj.get("text", "") if isinstance(type_obj, dict) else ""
                        summary_obj = p.get("editorialSummary", {})
                        summary = summary_obj.get("text", "") if isinstance(summary_obj, dict) else ""

                        all_places.append({
                            "name": name,
                            "address": p.get("formattedAddress", ""),
                            "phone": phone,
                            "website": p.get("websiteUri", ""),
                            "rating": str(p.get("rating", "")),
                            "category": category,
                            "summary": summary
                        })

                        if len(all_places) >= limit:
                            break

                    next_page_token = resp_json.get("nextPageToken")
                    if not next_page_token or len(all_places) >= limit:
                        break
                    
                    time.sleep(0.3)
                else:
                    console.print(f"[red]Google Places API Error ({resp.status_code}):[/red] {resp.text}")
                    break
            except Exception as e:
                console.print(f"[red]Google Places API request failed:[/red] {e}")
                break

        return all_places

    def _search_free_web(self, query: str, limit: int = 5) -> List[Dict]:
        results = []
        raw_results = self.search_web(query, limit=limit * 3)

        ignored_domains = [
            "yelp.", "tripadvisor.", "yellowpages.", "wikipedia.org", "facebook.com", 
            "instagram.com", "f6s.com", "clutch.co", "builtin.", "inven.ai", 
            "reddit.com", "linkedin.com", "zoominfo.com", "g2.com"
        ]

        for item in raw_results:
            title = item.get("title", "")
            href = item.get("href", "")
            snippet = item.get("snippet", "")

            domain = urlparse(href).netloc.lower()
            if any(ign in domain for ign in ignored_domains):
                continue

            clean_name = title.split(" - ")[0].split(" | ")[0].split(" : ")[0].strip()
            phone_match = re.search(r'(\+?[\d\s\-\(\)]{10,20}\d)', snippet)
            phone = phone_match.group(0).strip() if phone_match else ""

            results.append({
                "name": clean_name,
                "address": "Web Result",
                "phone": phone,
                "website": href,
                "rating": "",
                "category": "Business / Organization",
                "summary": snippet[:150]
            })

            if len(results) >= limit:
                break

        return results

    def scrape_and_analyze_company(self, website_url: str, company_name: str, base_summary: str = "") -> Dict[str, str]:
        """Fast concurrent crawl for emails, phones, and AI summary with tight timeouts."""
        emails = set()
        phones = set()
        extracted_text_chunks = []
        
        if website_url and website_url.startswith("http") and "duckduckgo.com" not in website_url:
            urls_to_check = [website_url]
            try:
                parsed = urlparse(website_url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                urls_to_check.append(urljoin(base_url, "/contact"))
            except Exception:
                pass

            email_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
            phone_pattern = re.compile(r'(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}')
            ignored_email_domains = ["duckduckgo.com", "google.com", "example.com", "w3.org", "sentry.io", "itgovernance.co.uk", "cloudflare.com"]

            for target_url in urls_to_check[:2]:
                try:
                    resp = self.session.get(target_url, timeout=2.5)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        
                        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
                        if meta_desc and meta_desc.get('content'):
                            extracted_text_chunks.append(meta_desc['content'].strip())

                        for a_tag in soup.select('a[href^="mailto:"]'):
                            email = a_tag['href'].replace('mailto:', '').split('?')[0].strip()
                            if email and "@" in email:
                                domain = email.split("@")[-1].lower()
                                if not any(d in domain for d in ignored_email_domains):
                                    emails.add(email)

                        for a_tag in soup.select('a[href^="tel:"]'):
                            tel = a_tag['href'].replace('tel:', '').strip()
                            if tel:
                                phones.add(tel)

                        for e in email_pattern.findall(resp.text):
                            ext = e.lower().split('.')[-1]
                            domain = e.split("@")[-1].lower()
                            if ext not in ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'css', 'js', 'html']:
                                if not any(d in domain for d in ignored_email_domains):
                                    emails.add(e)

                        if not phones:
                            for p in phone_pattern.findall(resp.text):
                                if len(re.sub(r'\D', '', p)) >= 10:
                                    phones.add(p.strip())
                except Exception:
                    continue

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
            if combined_text:
                clean_snippet = re.sub(r'\s+', ' ', combined_text).strip()
                ai_summary = clean_snippet[:160] + "..." if len(clean_snippet) > 160 else clean_snippet
            else:
                ai_summary = f"Commercial business specializing in {company_name} products & services."

        return {
            "email": ", ".join(list(emails)[:3]),
            "phone": ", ".join(list(phones)[:2]),
            "ai_summary": ai_summary
        }

    def find_linkedin_and_employees(self, company_name: str, role_keywords: str = "CEO OR Founder OR Director OR Owner") -> Dict[str, str]:
        """Search for company LinkedIn page and key employee listings."""
        linkedin_company = ""
        employees = []

        company_results = self.search_web(f'site:linkedin.com/company "{company_name}"', limit=1)
        if company_results:
            linkedin_company = company_results[0].get("href", "")

        people_results = self.search_web(f'site:linkedin.com/in "{company_name}" ({role_keywords})', limit=3)
        for p in people_results:
            title = p.get("title", "")
            link = p.get("href", "")
            clean_title = title.split(" - ")[0].split(" | ")[0].replace(" | LinkedIn", "").strip()
            if clean_title:
                employees.append(f"{clean_title} ({link})")

        return {
            "linkedin_company": linkedin_company,
            "employees": " \n".join(employees) if employees else "None found"
        }

    def enrich_single_company(self, comp: Dict, roles: str) -> Dict:
        """Worker function to enrich a single company concurrently."""
        name = comp.get("name", "Unknown")
        website = comp.get("website", "")
        summary = comp.get("summary", "")

        # 1. Scrape emails & AI summary
        analysis = self.scrape_and_analyze_company(website, name, base_summary=summary)
        comp["email"] = analysis.get("email", "")
        if not comp.get("phone"):
            comp["phone"] = analysis.get("phone", "")
        comp["ai_summary"] = analysis.get("ai_summary", "")

        # 2. LinkedIn & Employees
        social_info = self.find_linkedin_and_employees(name, role_keywords=roles)
        comp["linkedin_company"] = social_info.get("linkedin_company", "")
        comp["employees"] = social_info.get("employees", "")

        return comp

    def run(self, query: str, limit: int = 10, output_format: str = "csv", output_path: Optional[str] = None):
        console.print(f"\n[bold green]⚡ [Parallel AI Engine] Searching:[/bold green] [bold white]'{query}'[/bold white] (Limit: {limit})\n")

        ai_query_data = self.ai_enhance_query(query)
        search_term = ai_query_data.get("places_query", query)
        roles = ai_query_data.get("roles", "CEO OR Founder OR Director")

        # 1. Fetch places
        companies = self.search_companies(search_term, limit=limit)
        if not companies:
            console.print("[bold red]No companies found. Try a different search query.[/bold red]")
            return None

        console.print(f"[bold green]✔ Found {len(companies)} results.[/bold green] Running multi-threaded enrichment across {self.max_workers} parallel workers...\n")

        # 2. Multi-threaded parallel enrichment
        enriched_data = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Enriching {len(companies)} leads in parallel...", total=len(companies))

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

        self.display_summary_table(enriched_data[:10]) # Show top 10 preview
        console.print(f"\n[bold green]✔ Successfully saved {len(enriched_data)} AI-enriched leads to:[/bold green] [bold underline yellow]{os.path.abspath(output_path)}[/bold underline yellow]\n")
        return df

    def display_summary_table(self, data: List[Dict]):
        table = Table(title="AI-Enriched Company Leads Preview", show_header=True, header_style="bold magenta")
        table.add_column("Company Name", style="bold cyan", max_width=20)
        table.add_column("Phone", style="green", max_width=16)
        table.add_column("Email", style="yellow", max_width=22)
        table.add_column("AI Summary", style="white", max_width=30)
        table.add_column("LinkedIn", style="magenta", max_width=22)

        for row in data:
            table.add_row(
                str(row.get("name", "")),
                str(row.get("phone", "")),
                str(row.get("email", "")),
                str(row.get("ai_summary", "")),
                str(row.get("linkedin_company", ""))
            )

        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Multi-threaded AI Google Maps & Company Enrichment Bot")
    parser.add_argument("-q", "--query", type=str, help="Search query (e.g. 'AI Startups in Austin TX')")
    parser.add_argument("-l", "--limit", type=int, default=10, help="Maximum number of companies to extract (default: 10, up to 100)")
    parser.add_argument("-f", "--format", choices=["csv", "xlsx", "json"], default="csv", help="Export format (default: csv)")
    parser.add_argument("-o", "--output", type=str, help="Custom output file path")
    parser.add_argument("-w", "--workers", type=int, default=12, help="Number of concurrent worker threads (default: 12)")

    args = parser.parse_args()

    query = args.query
    if not query:
        console.print("\n[bold cyan]=== Multi-Threaded AI Company Enrichment Bot ===[/bold cyan]")
        query = console.input("[bold yellow]Enter your search query (e.g. 'Robotics companies in Boston'): [/bold yellow]").strip()
        if not query:
            query = "Robotics companies in Boston"

    bot = CompanyEnrichmentBot(max_workers=args.workers)
    bot.run(query=query, limit=args.limit, output_format=args.format, output_path=args.output)


if __name__ == "__main__":
    main()
