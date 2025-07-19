import pandas as pd
import numpy as np
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
import pickle
import os
import json


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


def prepare_sequence_data(df):
    sequences = []
    labels = []
    for (pdf_file, page), group in df.groupby(['pdf_file', 'page']):
        group = group.sort_values('rel_y')
        sequence_features = []
        sequence_labels = []
        for _, row in group.iterrows():
            features = extract_features(
                row['text'], row['font_size'], row['is_bold'], row['is_italic'],
                row['rel_y'], row['length'], row['num_words'], row['x'], row['color']
            )
            sequence_features.append(features)
            sequence_labels.append(row['heading_level'])
        if sequence_features:
            sequences.append(sequence_features)
            labels.append(sequence_labels)
    return sequences, labels


def train_crf_model(X_train, y_train):
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=True
    )
    crf.fit(X_train, y_train)
    return crf


def save_model(crf, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(crf, f)
    print(f"Model saved to {model_path}")


def main():
    print("=== Training CRF Model on Custom Data ===")
    train_path = "datasets/final_datasets/custom_train.csv"
    if not os.path.exists(train_path):
        print("Custom training CSV not found.")
        return
    train_df = pd.read_csv(train_path)
    print(f"Training samples: {len(train_df)}")
    X_train, y_train = prepare_sequence_data(train_df)
    print(f"Training sequences: {len(X_train)}")
    crf_model = train_crf_model(X_train, y_train)
    model_path = "datasets/final_datasets/crf_headings_model_custom.pkl"
    save_model(crf_model, model_path)
    print("=== Training Complete ===")


if __name__ == "__main__":
    main()
