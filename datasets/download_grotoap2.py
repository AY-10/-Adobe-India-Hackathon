import fitz  # PyMuPDF
import glob
import os
import requests
import zipfile
import pandas as pd
from io import BytesIO

GROTOAP2_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/grotoap2/grotoap2-updated.zip"
DATA_DIR = "datasets/grotoap2"
PROCESSED_CSV = "datasets/grotoap2_processed/grotoap2_enhanced.csv"
ZIP_PATH = os.path.join(DATA_DIR, "grotoap2-updated.zip")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("datasets/grotoap2_processed", exist_ok=True)

# Download GROTOAP2 zip file with streaming
if not os.path.exists(ZIP_PATH):
    print("Downloading GROTOAP2 dataset (this may take a while)...")
    with requests.get(GROTOAP2_URL, stream=True) as r:
        r.raise_for_status()
        with open(ZIP_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {ZIP_PATH}")
else:
    print(f"ZIP file already exists at {ZIP_PATH}")

# Extract GROTOAP2 zip file
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    zf.extractall(DATA_DIR)
print("Extracted GROTOAP2 dataset.")

# For demo: process a small subset (e.g., 10 PDFs)


def extract_features(text, font_size, is_bold, is_italic, rel_y, length, num_words, x, color):
    features = {
        'font_size': font_size,
        'is_bold': is_bold,
        'is_italic': is_italic,
        'rel_y': rel_y,
        'length': length,
        'num_words': num_words,
        'x': x,
        'color': color,
        'text_length': len(text),
        'starts_with_number': int(text[0].isdigit() if text else False),
        'ends_with_colon': int(text.endswith(':') if text else False),
        'all_caps': int(text.isupper() if text else False),
        'title_case': int(text.istitle() if text else False),
        'has_numbers': int(any(c.isdigit() for c in text) if text else False),
        'has_special_chars': int(any(c in '()[]{}' for c in text) if text else False),
        'font_size_large': int(font_size > 14),
        'font_size_medium': int(10 <= font_size <= 14),
        'font_size_small': int(font_size < 10),
        'position_top': int(rel_y < 0.2),
        'position_middle': int(0.2 <= rel_y <= 0.8),
        'position_bottom': int(rel_y > 0.8),
        'short_text': int(length < 20),
        'medium_text': int(20 <= length <= 50),
        'long_text': int(length > 50),
        'single_word': int(num_words == 1),
        'few_words': int(2 <= num_words <= 5),
        'many_words': int(num_words > 5)
    }
    return features


# For demo, process only 10 PDFs
pdf_files = glob.glob(os.path.join(DATA_DIR, "pdfs", "*.pdf"))[:10]
annotation_dir = os.path.join(DATA_DIR, "annotations")
rows = []

for pdf_path in pdf_files:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    ann_path = os.path.join(annotation_dir, base + ".csv")
    if not os.path.exists(ann_path):
        continue
    # Load annotation CSV (columns: page, text, heading_level)
    ann_df = pd.read_csv(ann_path)
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                line_text = " ".join([span["text"]
                                     for span in line["spans"]]).strip()
                if not line_text:
                    continue
                span = line["spans"][0]
                font_size = span["size"]
                is_bold = int("Bold" in span["font"])
                is_italic = int(
                    "Italic" in span["font"] or "Oblique" in span["font"])
                rel_y = line["bbox"][1] / page.rect.height
                length = len(line_text)
                num_words = len(line_text.split())
                x = line["bbox"][0]
                color = span.get("color", 0)
                # Label: match annotation by text and page
                ann = ann_df[(ann_df['page'] == page_num) & (
                    ann_df['text'].str.strip() == line_text.strip())]
                heading_level = ann['heading_level'].values[0] if not ann.empty else 'not_heading'
                feat = extract_features(
                    line_text, font_size, is_bold, is_italic, rel_y, length, num_words, x, color)
                row = {
                    "pdf_file": os.path.basename(pdf_path),
                    "page": page_num,
                    "text": line_text,
                    "font_size": font_size,
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                    "rel_y": rel_y,
                    "length": length,
                    "num_words": num_words,
                    "x": x,
                    "color": color,
                    "heading_level": heading_level
                }
                row.update(feat)
                rows.append(row)

pd.DataFrame(rows).to_csv(PROCESSED_CSV, index=False)
print(f"Saved processed GROTOAP2 data to {PROCESSED_CSV}")
