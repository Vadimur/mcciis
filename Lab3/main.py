#!/usr/bin/env python3
"""
Wikipedia Web Agent (single-file Flask app)

Features:
- Web GUI with a search field
- Uses the Wikipedia API (action=query) to search pages
- Returns a list of actual page links with a short description (extract)

How to run:
1. Install dependencies: pip install flask requests
2. Run: python wikipedia_web_agent.py
3. Open browser: http://127.0.0.1:5000

This file contains both the Flask backend and minimal frontend (Bootstrap + JS).
"""

from flask import Flask, render_template_string, request, jsonify
import requests

app = Flask(__name__)

# -------------------- Configuration --------------------
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
RESULT_LIMIT = 5
EXTRACT_CHARS = 500

# -------------------- Templates --------------------
INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Wikipedia Web Agent</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body class="bg-light">
    <div class="container py-5">
      <div class="card shadow-sm">
        <div class="card-body">
          <h1 class="card-title">Wikipedia Web Agent</h1>
          <p class="card-text">Enter any search query and receive a list of matching Wikipedia pages with short descriptions and links.</p>

          <form id="search-form" class="row g-2 mb-3">
            <div class="col-sm-9">
              <input id="q" name="q" class="form-control" placeholder="Search Wikipedia..." autocomplete="off">
            </div>
            <div class="col-sm-3 d-grid">
              <button id="search-btn" class="btn btn-primary">Search</button>
            </div>
          </form>

          <div id="results"></div>
        </div>
      </div>

      <footer class="mt-3 text-muted small">Powered by the <a href="https://www.mediawiki.org/wiki/API:Main_page" target="_blank">Wikipedia API</a></footer>
    </div>

    <script>
      const form = document.getElementById('search-form');
      const results = document.getElementById('results');

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const q = document.getElementById('q').value.trim();
        if (!q) return;
        results.innerHTML = `<div class="text-center py-4">Searching for <strong>${q}</strong> ...</div>`;

        try {
          const resp = await fetch('/search?q=' + encodeURIComponent(q));
          if (!resp.ok) throw new Error('Server error');
          const data = await resp.json();

          if (data.results.length === 0) {
            results.innerHTML = `<div class="alert alert-warning">No results for <strong>${q}</strong>.</div>`;
            return;
          }

            const html = data.results.map(r => `
                <div class="card mb-2">
                    <div class="card-body d-flex">
                        ${r.thumbnail ? `<img src="${r.thumbnail}" class="me-3 rounded" style="width:80px;height:80px;object-fit:cover;"/>` : `<div class='me-3 rounded bg-secondary' style='width:80px;height:80px;'></div>`}
                            <div>
                                <h5 class="card-title"><a href="${r.url}" target="_blank">${r.title}</a></h5>
                                <p class="card-text">${r.summary}</p>
                                <p class="card-text">${r.fact}</p>
                                <a class="btn btn-sm btn-outline-secondary" href="${r.url}" target="_blank">Open page</a>
                            </div>
                    </div>
                </div>
            `).join('');

          results.innerHTML = html;
        } catch (err) {
          results.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
        }
      });
    </script>
  </body>
</html>
"""

# -------------------- LLM Summarization --------------------

import openai
with open("secret.txt") as f:
    openai.api_key = f.read().strip()

def summarize_text(text):
    """Use an LLM to produce a short 2–3 sentence summary of the extracted Wiki text."""
    if not text:
        return ""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Summarize the text in 2-3 short sentences. "
                                                    "Add short interesting fact about the topic of the text at the end. Put one empty line before the fact."},
                      {"role": "user", "content": text}],
            max_tokens=500
        )
        print(resp)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print (e)
        return text[:150] + "..."

# -------------------- Helper: call Wikipedia API --------------------

def wiki_search(query, limit=RESULT_LIMIT, extract_chars=EXTRACT_CHARS):
    """
    Performs a search using Wikipedia's API and returns a list of dicts:
    {title, pageid, url, extract}
    """
    params = {
        'action': 'query',
        'format': 'json',
        'generator': 'search',
        'gsrsearch': query,
        'gsrlimit': limit,
        'prop': 'info|extracts|pageimages',
        'inprop': 'url',
        'piprop': 'thumbnail',
        'pithumbsize': 200,
        'exintro': True,
        'explaintext': True,
        'exchars': extract_chars,
    }

    try:
        r = requests.get(WIKIPEDIA_API, params=params, timeout=8, headers={"User-Agent": "WikipediaWebAgent/1.0 (test@example.com)"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        # Return empty list on failure; caller should handle errors
        return [], str(e)

    if 'query' not in data or 'pages' not in data['query']:
        return [], None

    pages = data['query']['pages']

    results = []
    # pages is a dict keyed by pageid, order is not guaranteed; sort by index if available
    pages_list = sorted(pages.values(), key=lambda p: p.get('index', 0))

    for p in pages_list:
        title = p.get('title')
        pageid = p.get('pageid')
        fullurl = p.get('fullurl') or f"https://en.wikipedia.org/?curid={pageid}"
        thumb = p.get('thumbnail', {}).get('source') if 'thumbnail' in p else None

        extract = p.get('extract') or ''
        summarized_text = summarize_text(extract)
        summary = summarized_text.split('\n\n')[0]
        fact = summarized_text.split('\n\n')[1]
        results.append({'title': title,
                       'pageid': pageid,
                       'url': fullurl,
                       'thumbnail': thumb,
                       'summary': summary,
                       'fact': fact,
                       'extract': (extract[:150] + '...') if len(extract) > 150 else extract}
                       )

    return results, None

# -------------------- Routes --------------------

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/search')
def search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'error': 'empty query', 'results': []}), 400

    results, err = wiki_search(q)
    if err is not None:
        return jsonify({'error': str(err), 'results': []}), 500

    return jsonify({'results': results})

# -------------------- Run --------------------

if __name__ == '__main__':
    app.run(debug=True)
