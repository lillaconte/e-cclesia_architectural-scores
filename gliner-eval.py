import pandas as pd
import numpy as np
from gliner import GLiNER
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data
df = pd.read_csv('output/sample.csv')

# Initialize models
models = {
    'gliner_multi-v2.1': GLiNER.from_pretrained('urchade/gliner_multi-v2.1'),
    'gliner_multi-v2.1-medieval-latin': GLiNER.from_pretrained('medieval-data/gliner_multi-v2.1-medieval-latin'),
    'gliner-multitask-large-v0.5': GLiNER.from_pretrained('knowledgator/gliner-multitask-large-v0.5')
}
labels = ["building/type", "building/part", "building/material"]

# Extract dictionary-based terms
def extract_dict_terms(row):
    terms = set()
    for col in ['autonomous_terms', 'associative_terms']:
        if pd.notna(row[col]):
            words = [w.strip().lower() for w in str(row[col]).split('|')]
            terms.update(words)
    return terms

# Extract GLiNER predictions
def extract_gliner_terms(text, model):
    entities = model.predict_entities(text, labels)
    return {ent['text'].lower() for ent in entities}

df['dict_terms'] = df.apply(extract_dict_terms, axis=1)

# Extract text for both columns
def get_text(row, text_col):
    text = str(row[text_col]) if pd.notna(row[text_col]) else ''
    return text

# Store results per model and text type
model_text_results = {
    model_name: {
        'lemmatized_text': [],
        'inscription_interpretive_cleaning': []
    }
    for model_name in models
}
model_common_counts = {
    model_name: {
        'lemmatized_text': 0,
        'inscription_interpretive_cleaning': 0
    }
    for model_name in models
}

for idx, row in df.iterrows():
    dict_terms = row['dict_terms']
    if not dict_terms:
        continue

    for text_col in ['lemmatized_text', 'inscription_interpretive_cleaning']:
        text = get_text(row, text_col)
        if not text:
            continue

        for model_name, model in models.items():
            gliner_terms = extract_gliner_terms(text, model)

            # Compare
            common = dict_terms & gliner_terms
            common_pct = (len(common) / len(dict_terms) * 100) if dict_terms else 0

            # Count inscriptions with at least 1 common term
            if len(common) > 0:
                model_common_counts[model_name][text_col] += 1

            # Precision, Recall, F1
            all_terms = dict_terms | gliner_terms
            y_true = [1 if term in dict_terms else 0 for term in all_terms]
            y_pred = [1 if term in gliner_terms else 0 for term in all_terms]
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            model_text_results[model_name][text_col].append({
                'index': idx,
                'dict_terms_count': len(dict_terms),
                'gliner_terms_count': len(gliner_terms),
                'common_count': len(common),
                'common_pct': common_pct,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'text': text,
                'autonomous_terms': row['autonomous_terms'],
                'associative_terms': row['associative_terms'],
                'gliner_terms': '|'.join(gliner_terms)
            })

# Aggregate metrics per model and text type
metrics = []
for model_name in models:
    for text_col in ['lemmatized_text', 'inscription_interpretive_cleaning']:
        results = model_text_results[model_name][text_col]
        if not results:
            continue
        df_model = pd.DataFrame(results)
        avg_precision = df_model['precision'].mean()
        avg_recall = df_model['recall'].mean()
        avg_f1 = df_model['f1'].mean()
        avg_common_pct = df_model['common_pct'].mean()
        avg_common_count = df_model['common_count'].mean()
        inscriptions_with_common = model_common_counts[model_name][text_col]

        metrics.append({
            'Model': model_name,
            'Text Type': text_col,
            'Avg Precision': avg_precision,
            'Avg Recall': avg_recall,
            'Avg F1': avg_f1,
            'Avg Common %': avg_common_pct,
            'Avg Common Count': avg_common_count,
            'Inscriptions with Common': inscriptions_with_common
        })

metrics_df = pd.DataFrame(metrics)

# Print a simple table
header = "{:<30} {:<30} {:<15} {:<15} {:<15} {:<15} {:<20} {:<25}"
row = "{:<30} {:<30} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f} {:<20.3f} {:<25}"

print(header.format(
    "model", "text type", "precision", "recall", "f1",
    "average common terms %", "average common terms count", "inscriptions with common terms"
))
print("-" * 170)
for _, row_data in metrics_df.iterrows():
    print(row.format(
        row_data['Model'], row_data['Text Type'], row_data['Avg Precision'],
        row_data['Avg Recall'], row_data['Avg F1'], row_data['Avg Common %'],
        row_data['Avg Common Count'], row_data['Inscriptions with Common']
    ))

# Save to CSV
metrics_df.to_csv('output/gliner_comparison_metrics_detailed.csv', index=False)




