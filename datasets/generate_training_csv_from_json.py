import os
import glob
import json
import fitz  # PyMuPDF
import pandas as pd

# Feature extraction function (same as in extract_outline.py)


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


INPUT_DIR = "input"
OUTPUT_DIR = "output"
TRAIN_CSV = "datasets/final_datasets/custom_train.csv"

os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)

# Helper: load outline from JSON


def load_outline(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    outline = {}
    for item in data.get("outline", []):
        # Normalize text for matching
        key = (item["text"].strip(), int(item["page"]))
        outline[key] = item["level"]
    return outline


rows = []

for pdf_path in glob.glob(os.path.join(INPUT_DIR, "*.pdf")):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(OUTPUT_DIR, base + ".json")
    if not os.path.exists(json_path):
        print(f"No JSON outline for {pdf_path}, skipping.")
        continue
    outline = load_outline(json_path)
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
                # Label: check if this line is a heading in the outline
                key = (line_text.strip(), page_num)
                heading_level = outline.get(key, "not_heading")
                feat = extract_features(
                    line_text, font_size, is_bold, is_italic, rel_y, length, num_words, x, color)
                row = {
                    "pdf_file": base + ".pdf",
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

# Save to CSV
pd.DataFrame(rows).to_csv(TRAIN_CSV, index=False)
print(f"Saved training data to {TRAIN_CSV}")
