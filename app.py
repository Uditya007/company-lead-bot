import os
import io
import csv
import json
import concurrent.futures
from flask import Flask, render_template, request, jsonify, Response
from bot import CompanyEnrichmentBot, GOOGLE_PLACES_API_KEY, GEMINI_API_KEY
import db

app = Flask(__name__)

@app.route("/")
def index():
    has_places_key = bool(GOOGLE_PLACES_API_KEY and GOOGLE_PLACES_API_KEY != "your_google_places_api_key_here")
    total_saved = db.get_total_enriched_count()
    return render_template("index.html", has_places_key=has_places_key, total_saved=total_saved)

@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "total_saved": db.get_total_enriched_count()
    })

@app.route("/api/clear-history", methods=["POST"])
def clear_history():
    db.clear_all_history()
    return jsonify({
        "success": True,
        "total_saved": 0
    })

@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    limit = int(data.get("limit", 10))
    only_new_listed = bool(data.get("onlyNewListed", False))
    only_no_website = bool(data.get("onlyNoWebsite", False))
    exclude_already_enriched = bool(data.get("excludeAlreadyEnriched", True))

    if not query:
        return jsonify({"error": "Search query is required"}), 400

    bot = CompanyEnrichmentBot(google_api_key=GOOGLE_PLACES_API_KEY, gemini_api_key=GEMINI_API_KEY, max_workers=14)

    try:
        # 1. AI Query optimization
        ai_query_info = bot.ai_enhance_query(query)
        search_query = ai_query_info.get("places_query", query)
        roles = ai_query_info.get("roles", "CEO OR Founder OR Director")

        # 2. Search companies with newly-listed, no-website, & deduplication filters
        companies = bot.search_companies(
            query=search_query,
            limit=limit,
            only_new_listed=only_new_listed,
            only_no_website=only_no_website,
            exclude_already_enriched=exclude_already_enriched
        )
        
        # 3. High-speed parallel enrichment using ThreadPoolExecutor
        enriched = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
            future_to_comp = {
                executor.submit(bot.enrich_single_company, comp, roles): comp for comp in companies
            }
            for future in concurrent.futures.as_completed(future_to_comp):
                try:
                    res = future.result()
                    enriched.append(res)
                except Exception:
                    pass

        # 4. Automatically save newly enriched leads into memory database
        if enriched:
            db.save_leads_batch(enriched, query_tag=query)

        return jsonify({
            "success": True,
            "query": query,
            "ai_optimized_query": search_query,
            "count": len(enriched),
            "duplicates_skipped": bot.last_duplicates_skipped,
            "total_in_memory": db.get_total_enriched_count(),
            "results": enriched
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/export-csv", methods=["POST"])
def export_csv():
    data = request.get_json() or {}
    results = data.get("results", [])
    
    if not results:
        return jsonify({"error": "No data to export"}), 400

    output = io.StringIO()
    fields = [
        "name", "category", "has_website", "is_new_listing", "rating", "user_rating_count",
        "phone", "email", "address", "website", "google_maps_uri",
        "linkedin_company", "employees", "ai_summary"
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in results:
        writer.writerow({
            "name": row.get("name", ""),
            "category": row.get("category", ""),
            "has_website": "Yes" if row.get("website") else "No (Website Opportunity)",
            "is_new_listing": "Yes (Newly Listed)" if row.get("is_new_listing") else "No",
            "rating": row.get("rating", ""),
            "user_rating_count": row.get("user_rating_count", 0),
            "phone": row.get("phone", ""),
            "email": row.get("email", ""),
            "address": row.get("address", ""),
            "website": row.get("website", ""),
            "google_maps_uri": row.get("google_maps_uri", ""),
            "linkedin_company": row.get("linkedin_company", ""),
            "employees": row.get("employees", ""),
            "ai_summary": row.get("ai_summary", "")
        })

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=verified_company_leads.csv"}
    )

if __name__ == "__main__":
    print("\n🌐 Starting Verified AI Company Finder Web Server on http://127.0.0.1:5050 ...")
    app.run(host="127.0.0.1", port=5050, debug=False)
