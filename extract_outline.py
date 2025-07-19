import os
import glob
import json
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = os.path.join("datasets", "final_datasets",
                          "rf_headings_model_custom.pkl")
TRAIN_CSV = os.path.join("datasets", "final_datasets", "custom_train.csv")
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# Feature extraction function (same as in training)


def extract_features(text, font_size, is_bold, is_italic, rel_y, length, num_words, x, color):
    features = [
        font_size,
        is_bold,
        is_italic,
        rel_y,
        length,
        num_words,
        x,
        color,
        len(text),
        int(text[0].isdigit() if text else False),
        int(text.endswith(":") if text else False),
        int(text.isupper() if text else False),
        int(text.istitle() if text else False),
        int(any(c.isdigit() for c in text) if text else False),
        int(any(c in '()[]{}' for c in text) if text else False),
        int(font_size > 14),
        int(10 <= font_size <= 14),
        int(font_size < 10),
        int(rel_y < 0.2),
        int(0.2 <= rel_y <= 0.8),
        int(rel_y > 0.8),
        int(length < 20),
        int(20 <= length <= 50),
        int(length > 50),
        int(num_words == 1),
        int(2 <= num_words <= 5),
        int(num_words > 5)
    ]
    return features

# If model is missing, retrain automatically


def train_rf_model():
    print("Training RandomForest model from CSV...")
    train_df = pd.read_csv(TRAIN_CSV)
    feature_cols = [
        'font_size', 'is_bold', 'is_italic', 'rel_y', 'length', 'num_words', 'x', 'color',
        'text_length', 'starts_with_number', 'ends_with_colon', 'all_caps', 'title_case',
        'has_numbers', 'has_special_chars', 'font_size_large', 'font_size_medium',
        'font_size_small', 'position_top', 'position_middle', 'position_bottom',
        'short_text', 'medium_text', 'long_text', 'single_word', 'few_words', 'many_words'
    ]
    X = train_df[feature_cols].values
    labels = train_df['heading_level'].astype(str)
    label_map = {l: i for i, l in enumerate(sorted(labels.unique()))}
    inv_label_map = {i: l for l, i in label_map.items()}
    y = labels.map(label_map).values
    clf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X, y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': clf, 'label_map': label_map,
                    'inv_label_map': inv_label_map}, f)
    print(f"Model trained and saved to {MODEL_PATH}")
    return clf, label_map, inv_label_map

# Load or train model


def load_rf_model(force_retrain=False):
    if force_retrain or not os.path.exists(MODEL_PATH):
        return train_rf_model()
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['label_map'], data['inv_label_map']

# Helper: extract lines and features from PDF


def extract_pdf_lines_and_features(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []
    features = []
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue  # skip images, etc.
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
                feat = extract_features(
                    line_text, font_size, is_bold, is_italic, rel_y, length, num_words, x, color
                )
                lines.append({
                    "text": line_text,
                    "page": page_num,
                })
                features.append(feat)
    return lines, features

# Helper: build outline from predictions


def build_outline(lines, preds, inv_label_map):
    outline = []
    title = None
    for line, pred in zip(lines, preds):
        label = inv_label_map[pred]
        if label == "not_heading":
            continue
        if label == "H1" and not title:
            title = line["text"]
        outline.append({
            "level": label,
            "text": line["text"],
            "page": line["page"]
        })
    if not title and outline:
        title = outline[0]["text"]
    return title, outline

# Main processing loop


def main(force_retrain=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, label_map, inv_label_map = load_rf_model(
        force_retrain=force_retrain)
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    for pdf_path in pdf_files:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Processing {pdf_path} ...")
        lines, feats = extract_pdf_lines_and_features(pdf_path)
        if not lines:
            print(f"No text found in {pdf_path}")
            continue
        preds = model.predict(feats)
        title, outline = build_outline(lines, preds, inv_label_map)
        output = {
            "title": title if title else "",
            "outline": outline
        }
        out_path = os.path.join(OUTPUT_DIR, base + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved outline to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="PDF Outline Extractor with RandomForest")
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain the model from CSV')
    args = parser.parse_args()
    main(force_retrain=args.retrain)
