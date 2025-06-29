import torch
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime
from markupsafe import Markup

year = datetime.now().year


app = Flask(__name__)

# Load dataframe dan BM25
df = pd.read_pickle('bert_embeddings.pkl')
with open('bm25_model.pkl', 'rb') as f:
    bm25 = pickle.load(f)

# Tambahkan kolom 'id' jika belum ada
if 'id' not in df.columns:
    df = df.reset_index().rename(columns={'index': 'id'})

# Convert date column to datetime if it's not already
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')

@app.route('/', methods=['GET', 'POST'])
def search():
    results_dict = []
    query = ""
    top_n = 10  # Default
    start_date = ""
    end_date = ""
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        top_n = int(request.form.get('top_n', 10))
        start_date = request.form.get('start_date', '').strip()
        end_date = request.form.get('end_date', '').strip()

        # Jika tidak ada query dan tidak ada filter tanggal, tidak melakukan apa-apa
        if not query and not start_date and not end_date:
            return render_template('index.html', query=query, results=[], top_n=top_n, 
                                 start_date=start_date, end_date=end_date, year=year)

        # Jika ada query, lakukan pencarian BM25 dulu
        if query:
            tokenized_query = query.split()
            scores = bm25.get_scores(tokenized_query)
            
            # Ambil semua hasil yang relevan (bukan hanya top N)
            all_indices = scores.argsort()[::-1]  # Sort descending
            all_scores = scores[all_indices]
            
            # Filter hasil berdasarkan skor minimum (opsional)
            # relevant_mask = all_scores > 0
            # all_indices = all_indices[relevant_mask]
            # all_scores = all_scores[relevant_mask]
            
            results = df.iloc[all_indices].copy()
            results['bm25_score'] = all_scores
        else:
            # Jika tidak ada query, ambil semua data
            results = df.copy()
            results['bm25_score'] = 0

        # Filter berdasarkan tanggal jika ada
        if start_date:
            start_dt = pd.to_datetime(start_date)
            results = results[results['date'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            results = results[results['date'] <= end_dt]

        if len(results) == 0:
            return render_template('index.html', query=query, results=[], top_n=top_n, 
                                 start_date=start_date, end_date=end_date, year=year, 
                                 error_message="Tidak ada berita ditemukan dalam rentang tanggal tersebut.")

        # Ambil top N hasil
        results = results.head(top_n)
        
        # Highlight text jika ada query
        if query:
            results['highlighted_text'] = results['text'].apply(lambda x: highlight_text(x[:300], query))
        else:
            results['highlighted_text'] = results['text'].apply(lambda x: x[:300] + '...' if len(x) > 300 else x)
        
        # Convert ke dictionary
        results_dict = results[['id', 'title', 'highlighted_text', 'bm25_score', 'date']].to_dict(orient='records')

    return render_template('index.html', query=query, results=results_dict, top_n=top_n, 
                         start_date=start_date, end_date=end_date, year=year)


# fungsi untuk menghighleight text
def highlight_text(text, query):
    terms = query.lower().split()
    for term in terms:
        text = text.replace(term, f"<mark>{term}</mark>")
        # juga capital-case dan title-case
        text = text.replace(term.capitalize(), f"<mark>{term.capitalize()}</mark>")
        text = text.replace(term.upper(), f"<mark>{term.upper()}</mark>")
    return Markup(text)  # biar tag HTML-nya gak di-escape

@app.route('/details/<int:doc_id>')
def details(doc_id):
    document = df[df['id'] == doc_id].iloc[0]
    # Format tanggal untuk tampilan yang lebih baik
    if pd.notna(document['date']):
        document['formatted_date'] = document['date'].strftime('%d %B %Y')
    else:
        document['formatted_date'] = 'Tanggal tidak tersedia'
    return render_template('details.html', document=document, year=year)

if __name__ == '__main__':
    app.run(debug=True, port=5006)
