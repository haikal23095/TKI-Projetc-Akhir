import torch
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime

year = datetime.now().year


app = Flask(__name__)

# Load dataframe dan BM25
df = pd.read_pickle('bert_embeddings.pkl')
with open('bm25_model.pkl', 'rb') as f:
    bm25 = pickle.load(f)

# Tambahkan kolom 'id' jika belum ada
if 'id' not in df.columns:
    df = df.reset_index().rename(columns={'index': 'id'})

@app.route('/', methods=['GET', 'POST'])
def search():
    results_dict = []
    query = ""
    top_n = 10  # Default
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        top_n = int(request.form.get('top_n', 10))

        # Tokenisasi query untuk BM25
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)

        # Ambil top N hasil dan skor
        top_indices = scores.argsort()[-top_n:][::-1]
        top_scores = scores[top_indices]
        results = df.iloc[top_indices].copy()
        results['bm25_score'] = top_scores
        results['highlighted_text'] = results['text'].apply(lambda x: highlight_text(x[:300], query))

        # Ambil title dan content
        results_dict = results[['id', 'title', 'highlighted_text', 'bm25_score']].to_dict(orient='records')


    return render_template('index.html', query=query, results=results_dict, top_n=top_n, year=year)

from markupsafe import Markup

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
    return render_template('details.html', document=document, year=year)

if __name__ == '__main__':
    app.run(debug=True, port=5006)
