# 🏢 Company & Lead Enrichment Bot

A Python bot that searches for companies, extracts their contact details (emails, phone numbers, physical address), and discovers their LinkedIn company profile and key employee listings.

---

## ⚡ Features

- **Google Maps / Places Integration**: Fetches verified business addresses, phone numbers, website URLs, and ratings.
- **Smart Web Search Fallback**: Works immediately even without API keys via search engine fallback.
- **Direct Email Extraction**: Crawls company homepages, `/contact`, and `/about` pages to extract public email addresses.
- **LinkedIn & Employee Discovery**: Finds official company LinkedIn profiles and key executives/employees (Founders, CEOs, Directors, Managers).
- **Multi-Format Export**: Save leads directly to `.csv`, `.xlsx` (Excel), or `.json`.
- **Rich CLI Interface**: Progress bars and formatted summary tables.

---

## 🚀 Quick Start

### 1. Activate the Virtual Environment

```bash
cd /Users/Uditya/.gemini/antigravity/scratch/company-lead-bot
source venv/bin/activate
```

### 2. (Optional) Configure Google Places API Key
For exact Google Maps addresses, copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```
Edit `.env`:
```env
GOOGLE_PLACES_API_KEY=your_actual_key_here
```

### 3. Run the Bot

#### Interactive Mode:
```bash
python bot.py
```

#### Command-Line Arguments:
```bash
# Search with a specific query and limit
python bot.py -q "Software companies in New York" -l 5

# Export directly to Excel (.xlsx)
python bot.py -q "Marketing agencies in London" -l 10 -f xlsx

# Export to JSON with custom file name
python bot.py -q "Design studios in Berlin" -f json -o my_leads.json
```

---

## 📊 Output Fields

The generated file will contain the following columns:
- `name`: Business or company name
- `address`: Physical address (from Google Maps or Web)
- `phone`: Contact phone number
- `website`: Company website URL
- `rating`: Google Maps rating
- `email`: Extracted public emails
- `linkedin_company`: Official LinkedIn company URL
- `employees`: Key employee names, roles, and profile links
