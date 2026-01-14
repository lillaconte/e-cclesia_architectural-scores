import json
import pandas as pd
import spacy
import numpy as np
from typing import Dict, Set
import os

# Configuration
INPUT_FILE = 'data/EDCS_text_cleaned_2022-09-12.json'
OUTPUT_FILE = 'output/edcs_architectural_scores.csv'
OUTPUT_HIGH_FILE = 'output/edcs_architectural_scores_gt50.csv'

# Date range
DATE_FROM = 399
DATE_TO = 1199

# Provinces to exclude
EXCLUDED_PROVINCES = [
    "Achaia", "Aegyptus", "Arabia", "Armenia", "Asia", "Cappadocia", 
    "Cilicia", "Creta et Cyrenaica", "Cyprus", "Dacia", "Dalmatia", 
    "Galatia", "Lycia et Pamphylia", "Macedonia", "Mesopotamia", 
    "Moesia inferior", "Moesia superior", "Palaestina", "Pannonia inferior", 
    "Pannonia superior", "Pontus et Bithynia", "Regnum Bospori", "Syria", "Thracia"
]

# Architectural vocabulary, including orthographic variants
AUTONOMOUS_TERMS = [
    "basilica", "baptisterium", "monasterium", "oratorium", "ecclesia", 
    "templum", "altare", "altarium", "tabernaculum", "ciborium", "ambo", 
    "ambonis", "confessio", "lauacrum", "pulpitum", "transenna", 
    "presbiterium", "presbyterium", "aedes", "aeclesia", "aeclesia", 
    "aecclesia", "aecclaesia", "basilicula"
]

ASSOCIATIVE_TERMS = [
    "fenestra", "porta", "murus", "domus", "tabula", "rotunda", "crux", 
    "pilarium", "podium", "aula", "columna", "arcus", "porticus", "absis", 
    "atrium", "uestibulum", "ualuae", "cancellus", "basis", "pauimentum", 
    "pictura", "musiuum", "cuncta", "liminare", 
    "fons", "tectum", "decorare", "ornare", "aedificare", "edificare", 
    "cooperire", "depingere", "tegere","tabulare", "volvere", "musivum", "struere", "pergula", "fastigium"
]

MATERIAL_TERMS = [
    "marmor", "metallum", "aurum", "argentum", "saxa"
]

# Score weights
WEIGHTS = {
    'term_count': 0.45,
    'cooccurrence': 0.25,
    'proximity': 0.20,
    'density': 0.10
}

MAX_DISTANCE = 20
PROXIMITY_DECAY_RATE = 0.1


def load_and_filter_data(filepath: str) -> pd.DataFrame:
    """Load JSON and filter by date range and provinces."""
    print(f"\n{'='*80}")
    print("STEP 1: LOADING AND FILTERING DATA")
    print(f"{'='*80}")
    print(f"Loading data from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"✓ Total inscriptions loaded: {len(df):,}")
    
    # Show available columns
    print(f"\nAvailable columns in dataset: {len(df.columns)}")
    print(f"Key columns: {', '.join(df.columns[:10].tolist())}...")
    
    # Filter by date range
    print(f"\nFiltering by date range: {DATE_FROM}-{DATE_TO}")
    print("  Checking date fields...")
    
    def is_in_date_range(row):
        try:
            date_from = pd.to_numeric(row.get('date_not_before', None), errors='coerce')
            date_to = pd.to_numeric(row.get('date_not_after', None), errors='coerce')
            
            if pd.isna(date_from) or pd.isna(date_to):
                return False
            
            # Check if date range overlaps with our target range
            return not (date_to < DATE_FROM or date_from > DATE_TO)
        except:
            return False
    
    df['in_date_range'] = df.apply(is_in_date_range, axis=1)
    before_date_filter = len(df)
    df = df[df['in_date_range']].copy()
    removed_by_date = before_date_filter - len(df)
    print(f"  ✓ After date filtering: {len(df):,} inscriptions")
    print(f"    (removed {removed_by_date:,} inscriptions outside date range)")
    
    # Filter by province
    print(f"\nFiltering by province (excluding {len(EXCLUDED_PROVINCES)} provinces)")
    print(f"  Excluded provinces: {', '.join(EXCLUDED_PROVINCES[:5])}...")
    before_province_filter = len(df)
    df = df[~df['province'].isin(EXCLUDED_PROVINCES)].copy()
    removed_by_province = before_province_filter - len(df)
    print(f"  ✓ After province filtering: {len(df):,} inscriptions")
    print(f"    (removed {removed_by_province:,} inscriptions from excluded provinces)")
    
    # Save filtered data to a secondary subset
    filtered_output = 'output/edcs_filtered_inscriptions.csv'
    df.to_csv(filtered_output, index=False, encoding='utf-8')
    print(f"\n✓ Filtered data saved to: {filtered_output}")

    # Reset index to avoid iteration issues
    df = df.reset_index(drop=True)
    
    # Show remaining provinces
    remaining_provinces = df['province'].value_counts()
    print(f"\nRemaining provinces ({len(remaining_provinces)}):")
    for prov, count in remaining_provinces.head(10).items():
        print(f"  - {prov}: {count:,} inscriptions")
    if len(remaining_provinces) > 10:
        print(f"  ... and {len(remaining_provinces) - 10} more provinces")
    
    return df


def lemmatize_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Lemmatize the interpretive text using spaCy."""
    print(f"\n{'='*80}")
    print("STEP 2: LEMMATIZING TEXTS")
    print(f"{'='*80}")
    print("\nLoading Latin spaCy model (la_core_web_md)...")
    try:
        nlp = spacy.load("la_core_web_md")
        print("✓ Model loaded successfully")
    except OSError:
        print("⚠ Latin model not found. Installing...")
        os.system("python -m spacy download la_core_web_md")
        nlp = spacy.load("la_core_web_md")
        print("✓ Model installed and loaded")
    
    print(f"\nProcessing {len(df):,} inscriptions...")
    print("Using field: 'clean_text_interpretive_word'")
    
    lemmatized_texts = []
    empty_texts = 0
    
    for i in range(len(df)):
        if i % 100 == 0 and i > 0:
            print(f"  Progress: {i:,}/{len(df):,} ({(i/len(df)*100):.1f}%)")
        
        row = df.iloc[i]
        text = row.get('clean_text_interpretive_word', '')
        
        if pd.isna(text) or not text:
            lemmatized_texts.append('')
            empty_texts += 1
            continue
        
        doc = nlp(str(text))
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
        ]
        lemmatized_texts.append(" ".join(lemmas))
        
        # Print first example
        if i == 0 and lemmas:
            print(f"\n--- Example: First inscription ---")
            print(f"  EDCS-ID: {row.get('EDCS-ID', 'N/A')}")
            print(f"  Original text: {text}")
            print(f"  Lemmatized: {' '.join(lemmas)}")
            print(f"  Word count: {len(text.split())} → {len(lemmas)} lemmas")
            print()
    
    df['lemmatized_text'] = lemmatized_texts
    print(f"\n✓ Lemmatization complete!")
    print(f"  - Processed: {len(df):,} inscriptions")
    print(f"  - Empty/missing texts: {empty_texts:,}")
    print(f"  - With content: {len(df) - empty_texts:,}")
    
    return df


def calculate_score(lemmatized_text: str, 
                   auto_terms: Set[str], 
                   assoc_terms: Set[str],
                   mate_terms: Set[str]) -> Dict:
    """Calculate architectural concentration score for a single inscription."""
    
    if pd.isna(lemmatized_text) or not lemmatized_text:
        return {
            'score': 0.0,
            'autonomous': [],
            'associative': [],
            'material': [],
            'n_autonomous': 0,
            'n_associative': 0,
            'n_material': 0
        }
    
    lemmas = str(lemmatized_text).split()
    
    # Find terms and their positions
    autonomous_found = []
    associative_found = []
    material_found = []
    auto_assoc_positions = []
    material_positions = []
    
    for idx, lemma in enumerate(lemmas):
        if lemma in auto_terms:
            autonomous_found.append(lemma)
            auto_assoc_positions.append(idx)
        elif lemma in assoc_terms:
            associative_found.append(lemma)
            auto_assoc_positions.append(idx)
        elif lemma in mate_terms:
            material_found.append(lemma)
            material_positions.append(idx)
    
    n_auto = len(autonomous_found)
    n_assoc = len(associative_found)
    n_mate = len(material_found)
    total_terms = n_auto + n_assoc
    
    
    # Material terms only have value if there is at least one other assoc or auto term
    include_material = (n_auto + n_assoc) > 0
    
    # Build positions used for proximity computation
    positions = list(auto_assoc_positions)
    if include_material:
        positions.extend(material_positions)
    
    total_terms = n_auto + n_assoc + (n_mate if include_material else 0)
    
    if total_terms == 0:
        return {
            'score': 0.0,
            'autonomous': [],
            'associative': [],
            'material': [],
            'n_autonomous': 0,
            'n_associative': 0,
            'n_material': 0
        }
    # Component 1: Term count (logarithmic scale)
    score_count = min(np.log1p(total_terms) / np.log1p(5), 1.0)
    
    # Component 2: Co-occurrence of terms
    if n_auto == 0:
        score_cooc = 0.0
    elif n_auto > 0 and n_assoc > 0:
        score_cooc = 1.0
    elif n_auto > 1:
        score_cooc = 0.6
    else:
        score_cooc = 0.3
    
    # Component 3: Proximity (average distance between consecutive terms)
    if len(positions) > 1:
        positions.sort()
        distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        avg_distance = np.mean(distances)
        score_prox = np.exp(-0.1 * avg_distance)
    else:
        score_prox = 0.0
    
    # Component 4: Density (terms per lemma)
    density = total_terms / len(lemmas)
    score_density = min(density / 0.15, 1.0)
    
    # Final weighted score (0-100)
    final_score = (
        WEIGHTS['term_count'] * score_count +
        WEIGHTS['cooccurrence'] * score_cooc +
        WEIGHTS['proximity'] * score_prox +
        WEIGHTS['density'] * score_density
    ) * 100
    return {
        'score': round(final_score, 2),
        'autonomous': autonomous_found,
        'associative': associative_found,
        'material': material_found,
        'n_autonomous': n_auto,
        'n_associative': n_assoc,
        'n_material': n_mate
    }


def process_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Process all inscriptions and calculate scores."""
    
    print(f"\n{'='*80}")
    print("STEP 3: CALCULATING ARCHITECTURAL SCORES")
    print(f"{'='*80}")
    
    auto_set = set(AUTONOMOUS_TERMS)
    assoc_set = set(ASSOCIATIVE_TERMS)
    mate_set = set(MATERIAL_TERMS)
    
    print(f"\nArchitectural vocabulary:")
    print(f"  - Autonomous terms: {len(AUTONOMOUS_TERMS)} ({', '.join(AUTONOMOUS_TERMS[:5])}...)")
    print(f"  - Associative terms: {len(ASSOCIATIVE_TERMS)} ({', '.join(ASSOCIATIVE_TERMS[:5])}...)")
    print(f"  - Material terms: {len(MATERIAL_TERMS)} ({', '.join(MATERIAL_TERMS[:5])}...)")
    
    print(f"\nScore calculation weights:")
    for component, weight in WEIGHTS.items():
        print(f"  - {component}: {weight} ({weight*100:.0f}%)")
    
    print(f"\nProcessing {len(df):,} inscriptions...")
    
    scores = []
    autonomous_lists = []
    associative_lists = []
    material_lists = []
    
    # Track statistics
    inscriptions_with_terms = 0
    total_auto_terms = 0
    total_assoc_terms = 0
    total_mate_terms = 0
    
    for idx in range(len(df)):
        if idx % 500 == 0:
            print(f"  Progress: {idx:,}/{len(df):,} ({(idx/len(df)*100):.1f}%)")
        
        row = df.iloc[idx]
        result = calculate_score(row['lemmatized_text'], auto_set, assoc_set, mate_set)
        
        scores.append(result['score'])
        autonomous_lists.append('|'.join(result['autonomous']) if result['autonomous'] else '')
        associative_lists.append('|'.join(result['associative']) if result['associative'] else '')
        material_lists.append('|'.join(result['material']) if result['material'] else '')
        
        if result['score'] > 0:
            inscriptions_with_terms += 1
            total_auto_terms += result['n_autonomous']
            total_assoc_terms += result['n_associative']
            total_mate_terms += result['n_material']
    
    df['arch_score'] = scores
    df['autonomous_terms'] = autonomous_lists
    df['associative_terms'] = associative_lists
    df['material_terms'] = material_lists
    
    print(f"\n✓ Processing complete!\n")
    print(f"Results summary:")
    print(f"  - Total inscriptions: {len(df):,}")
    print(f"  - With architectural terms: {inscriptions_with_terms:,} ({(inscriptions_with_terms/len(df)*100):.1f}%)")
    print(f"  - Without terms: {len(df) - inscriptions_with_terms:,} ({((len(df)-inscriptions_with_terms)/len(df)*100):.1f}%)")
    print(f"\nTerm frequency:")
    print(f"  - Total autonomous terms found: {total_auto_terms:,}")
    print(f"  - Total associative terms found: {total_assoc_terms:,}")
    print(f"  - Average autonomous per inscription (with terms): {(total_auto_terms/inscriptions_with_terms if inscriptions_with_terms > 0 else 0):.2f}")
    print(f"  - Average associative per inscription (with terms): {(total_assoc_terms/inscriptions_with_terms if inscriptions_with_terms > 0 else 0):.2f}")
    print(f"\nScore statistics:")
    print(f"  - Mean score: {df['arch_score'].mean():.2f}")
    print(f"  - Median score: {df['arch_score'].median():.2f}")
    print(f"  - Max score: {df['arch_score'].max():.2f}")
    print(f"  - Score > 30: {(df['arch_score'] > 30).sum():,} inscriptions")
    print(f"  - Score > 40: {(df['arch_score'] > 40).sum():,} inscriptions")
    print(f"  - Score > 50: {(df['arch_score'] > 50).sum():,} inscriptions")
    print(f"  - Score > 60: {(df['arch_score'] > 60).sum():,} inscriptions")
    print(f"  - Score > 70: {(df['arch_score'] > 70).sum():,} inscriptions")
    
    return df


def show_top_inscriptions(df: pd.DataFrame, n: int = 10) -> None:
    """Display top N highest scoring inscriptions."""
    
    print(f"\n{'='*80}")
    print(f"Top {n} inscriptions by architectural score:")
    print(f"{'='*80}\n")
    
    top = df.nlargest(n, 'arch_score')
    
    for idx, (_, row) in enumerate(top.iterrows(), 1):
        print(f"{idx}. Score: {row['arch_score']:.2f}")
        print(f"   ID: {row['EDCS-ID']}")
        print(f"   Province: {row.get('province', 'N/A')}")
        print(f"   Place: {row.get('place', 'N/A')}")
        print(f"   Date: {row.get('date_not_before', '?')} to {row.get('date_not_after', '?')}")
        print(f"   Autonomous: {row['autonomous_terms']}")
        print(f"   Associative: {row['associative_terms']}")
        print(f"   Text: {row.get('clean_text_interpretive_word', 'N/A')[:100]}...")
        print()


def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("ARCHITECTURAL VOCABULARY ANALYSIS")
    print("Latin Inscriptions (399-1199 CE)")
    print("="*80)
    
    # Create output directory if needed
    os.makedirs('output', exist_ok=True)
    
    # Load and filter data
    df = load_and_filter_data(INPUT_FILE)
    
    if len(df) == 0:
        print("\n ERROR: No inscriptions match the filtering criteria!")
        print("   Check your date range and province exclusions.")
        return
    
    # Lemmatize texts
    df = lemmatize_texts(df)
    
    # Calculate scores
    df_scored = process_corpus(df)
    
    # Show top results
    show_top_inscriptions(df_scored, n=10)
    
    # Save results
    print(f"\n{'='*80}")
    print("STEP 4: SAVING RESULTS")
    print(f"{'='*80}")
    print(f"\nSaving to: {OUTPUT_FILE}")
    print(f"  - Format: CSV (comma-separated)")
    print(f"  - Encoding: UTF-8")
    print(f"  - Rows: {len(df_scored):,}")
    print(f"  - Columns: {len(df_scored.columns)}")
    
    df_scored.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)  # Convert to MB
    print(f"\n✓ File saved successfully!")
    print(f"  - Size: {file_size:.2f} MB")

    # Save one subset per decimal of scores
    df_scored['score_bin'] = df_scored['arch_score'].apply(lambda x: int(x // 10) * 10)
    for score_bin in sorted(df_scored['score_bin'].unique()):
        subset = df_scored[df_scored['score_bin'] == score_bin].copy()
        subset_file = f"output/edcs_architectural_scores_{score_bin}_{score_bin+9}.csv"
        subset.to_csv(subset_file, index=False, encoding='utf-8')
        subset_size = os.path.getsize(subset_file) / (1024 * 1024)
        print(f"\n✓ Subset file saved: {subset_file}")
        print(f"  - Score range: {score_bin} to {score_bin + 9}")
        print(f"  - Rows: {len(subset):,}")
        print(f"  - Size: {subset_size:.2f} MB")

    # Save high-score subset (>50)
    high_df = df_scored[df_scored['arch_score'] > 50].copy()
    high_count = len(high_df)
    if high_count > 0:
        high_df.to_csv(OUTPUT_HIGH_FILE, index=False, encoding='utf-8')
        high_size = os.path.getsize(OUTPUT_HIGH_FILE) / (1024 * 1024)
        print(f"\n✓ High-score file saved: {OUTPUT_HIGH_FILE}")
        print(f"  - Rows: {high_count:,}")
        print(f"  - Size: {high_size:.2f} MB")
    else:
        print("\nNo inscriptions with arch_score > 50 found; no high-score CSV created.")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nColumns in output file:")
    print(f"  Original columns: {len(df.columns) - 4}")  # Minus the new ones
    print(f"  New columns added: 4")
    print(f"    1. lemmatized_text (lemmatized Latin text)")
    print(f"    2. arch_score (architectural concentration score 0-100)")
    print(f"    3. autonomous_terms (found autonomous terms, pipe-separated)")
    print(f"    4. associative_terms (found associative terms, pipe-separated)")
    
    print("\nArchitectural score distribution:")
    scores = pd.to_numeric(df_scored['arch_score'], errors='coerce').dropna()
    total_count = len(scores)
    if total_count == 0:
        print("  No scores available.")
    else:
        count_gt0 = (scores > 0).sum()
        count_gt50 = (scores > 50).sum()
        count_gt75 = (scores > 75).sum()

        print(f"  - Count: {total_count}")
        print(f"  - > 0: {count_gt0}")
        print(f"  - > 50: {count_gt50}")
        print(f"  - > 75: {count_gt75}")
        print(f"  - Mean: {scores.mean():.2f}")
        print(f"  - Std: {scores.std():.2f}")
        print(f"  - Min: {scores.min():.2f}")
        print(f"  - Median: {scores[scores!=0].median():.2f}") # median without '0' values because there are too many
        print(f"  - Max: {scores.max():.2f}")
    
    print(f"\n{'='*80}")
    print("✓ ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()