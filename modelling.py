import json
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertModel, AutoModel
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

df = pd.read_csv('output_with_content.csv')

df.rename(columns={'category;': 'category'}, inplace=True)

df['text'] = df['content']  # Ganti ini agar tetap konsisten dengan nama sebelumnya

# Inisialisasi Stopword Remover dan Stemmer
stop_factory = StopWordRemoverFactory()
stop_remover = stop_factory.create_stop_word_remover()

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def preprocess(text):
    if isinstance(text, list):
        text = ' '.join(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Hapus tanda baca
    text = stop_remover.remove(text)  # Hapus stopword
    text = stemmer.stem(text)  # Stemming
    return text

tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")

def bert_encode(texts, batch_size=32, max_length=512):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding BERT"):
        batch_texts = [' '.join(text) if isinstance(text, list) else text for text in texts[i:i+batch_size]]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoded)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

df['text'] = df['text'].fillna('').astype(str)  # pastikan tidak ada NaN
df['bert_embedding'] = bert_encode(df['text'].tolist(), batch_size=32, max_length=128)

# Simpan embedding
df.to_pickle('bert_embeddings.pkl')

# PROSES BM25
# =========================
df['clean_text'] = df['text'].apply(preprocess)
df['bm25_tokens'] = df['clean_text'].apply(lambda x: x.split())
bm25 = BM25Okapi(df['bm25_tokens'].tolist())

# Simpan BM25
with open('bm25_model.pkl', 'wb') as f:
    pickle.dump(bm25, f)

print("✅ Proses selesai! Data siap digunakan untuk search engine.")