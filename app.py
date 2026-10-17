import os
import io
import csv
import json
import concurrent.futures
from flask import Flask, render_template, request, jsonify, Response
from bot import CompanyEnrichmentBot, GOOGLE_PLACES_API_KEY, GEMINI_API_KEY

app = Flask(__name__)

@app.route("/")
def index():
    has_places_key = bool(GOOGLE_PLACES_API_KEY and GOOGLE_PLACES_API_KEY != "your_google_places_api_key_here")
    return render_template("index.html", has_places_key=has_places_key)

@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    limit = int(data.get("limit", 10))

    if not query:
        return jsonify({"error": "Search query is required"}), 400

    bot = CompanyEnrichmentBot(google_api_key=GOOGLE_PLACES_API_KEY, gemini_api_key=GEMINI_API_KEY, max_workers=14)

    try:
        # 1. AI Query optimization
        ai_query_info = bot.ai_enhance_query(query)
        search_query = ai_query_info.get("places_query", query)
        roles = ai_query_info.get("roles", "CEO OR Founder OR Director")

        # 2. Search companies (Maps or Web)
        companies = bot.search_companies(search_query, limit=limit)
        
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

        return jsonify({
            "success": True,
            "query": query,
            "ai_optimized_query": search_query,
            "count": len(enriched),
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
    writer = csv.DictWriter(output, fieldnames=["name", "category", "ai_summary", "address", "phone", "email", "website", "rating", "linkedin_company", "employees"])
    writer.writeheader()
    for row in results:
        writer.writerow({
            "name": row.get("name", ""),
            "category": row.get("category", ""),
            "ai_summary": row.get("ai_summary", ""),
            "address": row.get("address", ""),
            "phone": row.get("phone", ""),
            "email": row.get("email", ""),
            "website": row.get("website", ""),
            "rating": row.get("rating", ""),
            "linkedin_company": row.get("linkedin_company", ""),
            "employees": row.get("employees", "")
        })

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=ai_enriched_company_leads.csv"}
    )

if __name__ == "__main__":
    print("\n🌐 Starting High-Speed AI Company Finder Web Server on http://127.0.0.1:5050 ...")
    app.run(host="127.0.0.1", port=5050, debug=False)
