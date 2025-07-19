import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import os

TRAIN_PATH = "datasets/final_datasets/custom_train.csv"
MODEL_PATH = "datasets/final_datasets/rf_headings_model_custom.pkl"

# Load data
train_df = pd.read_csv(TRAIN_PATH)

# Features to use (same as in extract_features)
feature_cols = [
    'font_size', 'is_bold', 'is_italic', 'rel_y', 'length', 'num_words', 'x', 'color',
    'text_length', 'starts_with_number', 'ends_with_colon', 'all_caps', 'title_case',
    'has_numbers', 'has_special_chars', 'font_size_large', 'font_size_medium',
    'font_size_small', 'position_top', 'position_middle', 'position_bottom',
    'short_text', 'medium_text', 'long_text', 'single_word', 'few_words', 'many_words'
]

X = train_df[feature_cols].values
# Encode labels
labels = train_df['heading_level'].astype(str)
label_map = {l: i for i, l in enumerate(sorted(labels.unique()))}
inv_label_map = {i: l for l, i in label_map.items()}
y = labels.map(label_map).values

# Train/test split (optional, here we use all for training)
clf = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X, y)

# Save model and label map
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': clf, 'label_map': label_map,
                'inv_label_map': inv_label_map}, f)
print(f"Model saved to {MODEL_PATH}")

# Print training report
preds = clf.predict(X)
print(classification_report(y, preds, target_names=[
      inv_label_map[i] for i in range(len(inv_label_map))]))
