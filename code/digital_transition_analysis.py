import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "bertopic_model"
METADATA_FILE = "metadata_map.json"
OUTPUT_DIR = "digital_transition_analysis"

# Transition Years (Start of Digital Format)
TRANSITION_YEARS = {
    "Anclajes": 2016,
    "Auster": 2018,
    "Cuadernos de Literatura": 2020,
    "Cuadernos del Cilha": 2011,
    "Orbis Tertius": 2015,
    "Revista de Culturas y Literaturas Comparadas": 2015,
    "Saga. Revista de Letras": 2014
}

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[created] {OUTPUT_DIR}/")

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

try:
    topic_model = BERTopic.load(MODEL_PATH)
    topics = topic_model.topics_
except Exception as e:
    print(f"Error loading model: {e}")
    topics = [] 


with open(METADATA_FILE, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# ============================================================================
# CLASSIFY DOCUMENTS (PRINT VS DIGITAL)
# ============================================================================


data = []
skipped_count = 0

# Normalize journal names in transition dict for easier matching
normalized_transitions = {k.lower().strip(): v for k, v in TRANSITION_YEARS.items()}

for filename, info in metadata.items():
    # Get info
    journal = info.get("journal", "Unknown")
    year_str = info.get("publication_year")
    
    # Skip if missing info
    if not journal or not year_str:
        skipped_count += 1
        continue
        
    try:
        year = int(year_str)
    except:
        skipped_count += 1
        continue
        

# Load topic assignments file to get reliable filename -> topic mapping
TOPIC_ASSIGNMENTS_FILE = "topic_assignments.xlsx"

try:
    df_assignments = pd.read_excel(TOPIC_ASSIGNMENTS_FILE)
    if 'Topic' not in df_assignments.columns:
        df_assignments.rename(columns=lambda x: x.capitalize(), inplace=True)
    
    
    for _, row in df_assignments.iterrows():
        filename = row['Filename']
        topic = row['Topic']
        
        # Get metadata (journal/year)
        meta = metadata.get(filename, {})
        journal = meta.get("journal", row.get('Journal', 'Unknown'))
        year_val = meta.get("publication_year", row.get('Year', 0))
        
        try:
            year = int(year_val)
        except:
            continue
            
        # Normalize journal for matching
        j_norm = journal.lower().strip()
        
        # Match with transition years
        transition_year = None
        
        # Direct lookup
        if j_norm in normalized_transitions:
            transition_year = normalized_transitions[j_norm]
        else:
            # Try partial match
            for key, val in normalized_transitions.items():
                if key in j_norm or j_norm in key:
                    transition_year = val
                    break
        
        if transition_year:
            fmt = "Digital" if year >= transition_year else "Print"
            data.append({
                "filename": filename,
                "topic": topic,
                "journal": journal,
                "year": year,
                "format": fmt,
                "transition_year": transition_year
            })
            
except Exception as e:
    print(f"Error processing assignments: {e}")
    exit()

df = pd.DataFrame(data)

# ============================================================================
# ANALYSIS: GLOBAL SHIFT
# ============================================================================


# Calculate percentages
print_counts = df[df['format']=='Print']['topic'].value_counts(normalize=True) * 100
digital_counts = df[df['format']=='Digital']['topic'].value_counts(normalize=True) * 100

# Combine into dataframe
topics_list = sorted(list(set(df['topic'].unique()) - {-1})) 
topics_list_all = sorted(df['topic'].unique())

shift_data = []
for t in topics_list_all:
    p_pct = print_counts.get(t, 0)
    d_pct = digital_counts.get(t, 0)
    diff = d_pct - p_pct
    
    shift_data.append({
        "Topic": t,
        "Print_Pct": p_pct,
        "Digital_Pct": d_pct,
        "Shift": diff,
        "Growth_Factor": d_pct / p_pct if p_pct > 0 else np.nan
    })

df_shift = pd.DataFrame(shift_data)
df_shift.to_excel(os.path.join(OUTPUT_DIR, "print_vs_digital_topics.xlsx"), index=False)

# ============================================================================
# VISUALIZATION
# ============================================================================

# 1. Side-by-Side Bar Chart
plt.figure(figsize=(14, 8))
melted = df_shift[df_shift['Topic'] != -1].melt(id_vars=['Topic'], value_vars=['Print_Pct', 'Digital_Pct'], var_name='Format', value_name='Percentage')
sns.barplot(data=melted, x='Topic', y='Percentage', hue='Format', palette=['#3498db', '#e74c3c'])
plt.title('Topic Distribution: Print vs. Digital Era', fontsize=15, fontweight='bold')
plt.ylabel('Percentage of Documents')
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "print_vs_digital_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()


# 2. Shift Chart (Waterfall-like)
plt.figure(figsize=(12, 6))
sns.barplot(data=df_shift[df_shift['Topic'] != -1], x='Topic', y='Shift', palette='RdBu_r')
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Net Shift in Topic Prevalence (Digital % - Print %)', fontsize=15, fontweight='bold')
plt.ylabel('Percentage Point Change')
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "topic_shift_magnitude.png"), dpi=300, bbox_inches='tight')
plt.close()


