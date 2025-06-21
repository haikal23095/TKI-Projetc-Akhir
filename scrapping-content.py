import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import csv

# Baca file CSV
df = pd.read_csv("indonesian-news-title2.csv", encoding="ISO-8859-1")

# Hapus baris yang tidak punya URL
df = df.dropna(subset=['url'])

# Fungsi untuk ambil isi artikel
def get_article_text(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, 'html.parser')

        # Untuk situs Detik.com, konten biasanya di tag <div class="detail__body-text">
        content_divs = soup.find_all("div", class_="detail__body-text")
        paragraphs = [p.get_text(strip=True) for div in content_divs for p in div.find_all("p")]

        return ' '.join(paragraphs)
    except Exception as e:
        print(f"Gagal mengambil {url}: {e}")
        return ""

# Buat kolom baru 'content' yang berisi isi artikel
df['content'] = df['url'].apply(get_article_text)

# Simpan ke file baru dengan delimiter koma
df.to_csv("output_with_content.csv", index=False, encoding='utf-8-sig', sep=',', quoting=csv.QUOTE_ALL, quotechar='"')
