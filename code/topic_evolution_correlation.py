import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from scipy.stats import linregress


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "bertopic_model"
METADATA_FILE = "metadata_map.json"
TOPIC_ASSIGNMENTS_FILE = "topic_assignments.xlsx"  
OUTPUT_DIR = "evolution_analysis"

# Cultural terms to track
CULTURAL_TERMS = ["cultura", "social", "politico", "cultural"]

# Number of top words to display
TOP_N_WORDS = 15

# Number of time periods
N_TIME_PERIODS = 9

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"\n[created] {OUTPUT_DIR}/")

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

topic_model = BERTopic.load(MODEL_PATH)
topics = topic_model.topics_


# Load metadata
with open(METADATA_FILE, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Try to load topic assignments if available
try:
    assignments_df = pd.read_excel(TOPIC_ASSIGNMENTS_FILE)
    filenames = assignments_df['filename'].tolist()
    years = pd.to_datetime(assignments_df['year'])
    journals = assignments_df['journal'].tolist()
    print(f"✓ Loaded topic assignments from Excel")
except:
    print("  Note: Could not load topic_assignments.xlsx")
    print("  Using metadata file to reconstruct...")
    # Reconstruct from metadata (assuming same order as model)
    filenames = list(metadata.keys())
    years = []
    journals = []
    for fn in filenames:
        info = metadata.get(fn, {})
        y = info.get("publication_year")
        years.append(pd.to_datetime(f"{int(y)}-01-01") if y else pd.NaT)
        journals.append(info.get("journal", "Unknown"))
    years = pd.Series(years)

# ============================================================================
# GET ORIGINAL TOPIC REPRESENTATIONS
# ============================================================================


unique_topics = sorted([t for t in set(topics) if t != -1])

# Store original representations
original_representations = {}
for topic_id in unique_topics:
    words = topic_model.get_topic(topic_id)
    if words:
        original_representations[topic_id] = {
            'words': [(w, score) for w, score in words[:TOP_N_WORDS]],
            'word_list': [w for w, score in words[:TOP_N_WORDS]],
            'count': sum(1 for t in topics if t == topic_id)
        }


# Display original topics
print("ORIGINAL TOPIC REPRESENTATIONS (from trained model)")
for topic_id in sorted(original_representations.keys()):
    rep = original_representations[topic_id]
    words_str = ", ".join(rep['word_list'][:10])
    print(f"Topic {topic_id:2d} ({rep['count']:3d} docs): {words_str}")

# ============================================================================
# IDENTIFY WHICH TOPICS CONTAIN CULTURAL TERMS
# ============================================================================

print("CULTURAL TERMS IN ORIGINAL TOPICS")

topics_with_terms = {term: [] for term in CULTURAL_TERMS}

for topic_id in sorted(original_representations.keys()):
    word_list = original_representations[topic_id]['word_list']
    terms_present = []
    
    for term in CULTURAL_TERMS:
        if term in word_list:
            topics_with_terms[term].append(topic_id)
            rank = word_list.index(term) + 1
            # Get score
            score = [s for w, s in original_representations[topic_id]['words'] if w == term][0]
            terms_present.append(f"'{term}' (rank {rank}, score {score:.3f})")
    
    if terms_present:
        print(f"Topic {topic_id}: {', '.join(terms_present)}")

print(f"\nSummary:")
for term in CULTURAL_TERMS:
    n_topics = len(topics_with_terms[term])
    pct = (n_topics / len(original_representations)) * 100
    print(f"  '{term}': appears in {n_topics} topics ({pct:.1f}%)")
    print(f"    Topics: {topics_with_terms[term]}")

# ============================================================================
# CREATE TIME PERIODS
# ============================================================================


# Filter valid timestamps
valid_indices = [i for i, ts in enumerate(years) if pd.notna(ts)]
valid_years = years[valid_indices]

# Create time bins with manually defined n_time_periods (9), with equal document counts
time_bins = pd.qcut(valid_years, q=N_TIME_PERIODS, duplicates='drop')

# Get bin ranges
bin_ranges = {}
for idx, period in enumerate(time_bins.cat.categories):
    period_label = f"Period_{idx + 1}"
    bin_ranges[period_label] = {
        'start': period.left.year,
        'end': period.right.year,
        'label': f"{period.left.year}-{period.right.year}"
    }

print(f"✓ Created {len(bin_ranges)} time periods:")
for period, info in sorted(bin_ranges.items()):
    period_count = sum(1 for b in time_bins if time_bins.cat.categories.get_loc(b) == int(period.split('_')[1]) - 1)
    print(f"  {period}: {info['label']} ({period_count} docs)")

# Create period assignment for each document
doc_periods = {}
for idx, (valid_idx, period) in enumerate(zip(valid_indices, time_bins)):
    period_idx = time_bins.cat.categories.get_loc(period)
    period_label = f"Period_{period_idx + 1}"
    doc_periods[valid_idx] = period_label

# ============================================================================
# ANALYSIS 1: TOPIC DISTRIBUTION OVER TIME
# ============================================================================

print("ANALYSIS 1: TOPIC DISTRIBUTION OVER TIME")


distribution_data = []

for period_name in sorted(bin_ranges.keys()):
    period_info = bin_ranges[period_name]
    print(f"\n{period_info['label']}:")
    
    # Get documents in this period
    period_doc_indices = [i for i, p in doc_periods.items() if p == period_name]
    period_topics = [topics[i] for i in period_doc_indices]
    
    # Count documents per topic
    for topic_id in sorted(original_representations.keys()):
        count = sum(1 for t in period_topics if t == topic_id)
        pct = (count / len(period_doc_indices)) * 100 if len(period_doc_indices) > 0 else 0
        
        distribution_data.append({
            'period': period_name,
            'year_range': period_info['label'],
            'topic_id': topic_id,
            'n_docs': count,
            'percentage': pct
        })
        
        if count > 0:
            print(f"  Topic {topic_id:2d}: {count:3d} docs ({pct:5.1f}%)")

# Save distribution data
distribution_df = pd.DataFrame(distribution_data)
distribution_df.to_excel(os.path.join(OUTPUT_DIR, "01_topic_distribution_over_time.xlsx"), index=False)


# Visualization: Stacked area chart
pivot_dist = distribution_df.pivot(index='year_range', columns='topic_id', values='percentage')
pivot_dist = pivot_dist.reindex([bin_ranges[p]['label'] for p in sorted(bin_ranges.keys())])

fig, ax = plt.subplots(figsize=(14, 8))
pivot_dist.plot(kind='area', stacked=True, ax=ax, alpha=0.7, 
                colormap='tab10', linewidth=0)
ax.set_xlabel('Time Period', fontsize=13, fontweight='bold')
ax.set_ylabel('Percentage of Documents', fontsize=13, fontweight='bold')
ax.set_title('Topic Distribution Over Time', fontsize=15, fontweight='bold', pad=20)
ax.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_topic_distribution_stacked.png"), dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# ANALYSIS 2: CULTURAL TERMS - TOPIC PREVALENCE OVER TIME
# ============================================================================


print("ANALYSIS 2: CULTURAL TERMS - Topic Prevalence Over Time")

cultural_prevalence_data = []

for term in CULTURAL_TERMS:
    print(f"\n'{term}' (present in topics: {topics_with_terms[term]})")
    
    for period_name in sorted(bin_ranges.keys()):
        period_info = bin_ranges[period_name]
        
        # Get documents in this period
        period_doc_indices = [i for i, p in doc_periods.items() if p == period_name]
        period_topics = [topics[i] for i in period_doc_indices]
        
        # Count documents in topics that contain this term
        docs_in_cultural_topics = sum(1 for t in period_topics if t in topics_with_terms[term])
        total_docs_in_period = len(period_doc_indices)
        
        pct = (docs_in_cultural_topics / total_docs_in_period) * 100 if total_docs_in_period > 0 else 0
        
        print(f"  {period_info['label']}: {docs_in_cultural_topics}/{total_docs_in_period} docs ({pct:.1f}%)")
        
        cultural_prevalence_data.append({
            'term': term,
            'period': period_name,
            'year_range': period_info['label'],
            'docs_in_topics_with_term': docs_in_cultural_topics,
            'total_docs': total_docs_in_period,
            'percentage': pct
        })

# Save data
prevalence_df = pd.DataFrame(cultural_prevalence_data)
prevalence_df.to_excel(os.path.join(OUTPUT_DIR, "02_cultural_terms_prevalence.xlsx"), index=False)


# Visualization: Line graph
plt.figure(figsize=(12, 7))

for term in CULTURAL_TERMS:
    term_data = prevalence_df[prevalence_df['term'] == term].sort_values('period')
    plt.plot(term_data['year_range'], term_data['percentage'], 
             marker='o', linewidth=2.5, markersize=10, label=f"'{term}'")

plt.xlabel('Time Period', fontsize=13, fontweight='bold')
plt.ylabel('% of Documents in Topics Containing Term', fontsize=13, fontweight='bold')
plt.title('Cultural Studies Vocabulary: Document Distribution Over Time', 
          fontsize=15, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='best', framealpha=0.9)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.ylim(0, max(prevalence_df['percentage']) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_cultural_terms_prevalence_graph.png"), dpi=300, bbox_inches='tight')
plt.close()


# ============================================================================
# ANALYSIS 3: INDIVIDUAL TOPIC TRENDS (FOR TOPICS WITH CULTURAL TERMS)
# ============================================================================

print("ANALYSIS 3: INDIVIDUAL TOPIC TRENDS (Topics with Cultural Terms)")

# Identify topics that contain at least one cultural term
cultural_topics = set()
for term in CULTURAL_TERMS:
    cultural_topics.update(topics_with_terms[term])

print(f"Topics containing at least one cultural term: {sorted(cultural_topics)}")

topic_trends_data = []

for topic_id in sorted(cultural_topics):
    print(f"\nTopic {topic_id}:")
    rep = original_representations[topic_id]
    print(f"  Words: {', '.join(rep['word_list'][:10])}")
    
    # Which cultural terms does it contain?
    terms_in_topic = [term for term in CULTURAL_TERMS if topic_id in topics_with_terms[term]]
    terms_str = ', '.join([f"'{t}'" for t in terms_in_topic]) if terms_in_topic else "None"
    print(f"  Contains: {terms_str}")
    
    for period_name in sorted(bin_ranges.keys()):
        period_info = bin_ranges[period_name]
        
        # Count docs in this topic in this period
        period_doc_indices = [i for i, p in doc_periods.items() if p == period_name]
        period_topics = [topics[i] for i in period_doc_indices]
        
        count = sum(1 for t in period_topics if t == topic_id)
        pct = (count / len(period_doc_indices)) * 100 if len(period_doc_indices) > 0 else 0
        
        print(f"    {period_info['label']}: {count:3d} docs ({pct:5.1f}%)")
        
        topic_trends_data.append({
            'topic_id': topic_id,
            'period': period_name,
            'year_range': period_info['label'],
            'n_docs': count,
            'percentage': pct,
            'cultural_terms': ', '.join(terms_in_topic)
        })

# Save data
trends_df = pd.DataFrame(topic_trends_data)
trends_df.to_excel(os.path.join(OUTPUT_DIR, "03_cultural_topics_trends.xlsx"), index=False)


# Visualization: Line graph for each cultural topic
plt.figure(figsize=(14, 8))

for topic_id in sorted(cultural_topics):
    topic_data = trends_df[trends_df['topic_id'] == topic_id].sort_values('period')
    plt.plot(topic_data['year_range'], topic_data['percentage'], 
             marker='o', linewidth=2, markersize=8, label=f"Topic {topic_id}")

plt.xlabel('Time Period', fontsize=13, fontweight='bold')
plt.ylabel('% of Documents', fontsize=13, fontweight='bold')
plt.title('Trends for Topics Containing Cultural Studies Vocabulary', 
          fontsize=15, fontweight='bold', pad=20)
plt.legend(fontsize=10, loc='best', framealpha=0.9)
plt.grid(axis='both', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_cultural_topics_trends_graph.png"), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# ANALYSIS 4: STATISTICAL TREND ANALYSIS (REGRESSION)
# ============================================================================

print("ANALYSIS 4: STATISTICAL TREND ANALYSIS (Topic vs. Time)")

# 1. Setup Data for Regression
# We analyze ALL topics (excluding outliers for trend clarity)
unique_topics = sorted([t for t in set(topics) if t != -1])
periods = sorted(bin_ranges.keys())
x_values = range(len(periods)) # Time steps: [0, 1, 2, 3...]

trend_results = []

for topic_id in unique_topics:
    # Get keywords for context
    rep = original_representations[topic_id]
    words_str = ", ".join(rep['word_list'][:5])
    
    # Calculate Y values (Percentage of documents in each period)
    y_values = []
    
    for period_name in periods:
        period_doc_indices = [i for i, p in doc_periods.items() if p == period_name]
        period_topics = [topics[i] for i in period_doc_indices]
        total_docs = len(period_doc_indices)
        
        count = sum(1 for t in period_topics if t == topic_id)
        # We use percentage to normalize against period size
        pct = (count / total_docs) * 100 if total_docs > 0 else 0
        y_values.append(pct)
    
    # 2. Perform Linear Regression (The "General Overall Trend")
    # Slope: How much the % changes per period on average
    # P-value: Probability that this trend is random luck
    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
    
    # 3. Classify Trend based on Slope and Significance
    # We require p < 0.05 to call it a "Significant" trend
    if p_value < 0.05:
        if slope > 0:
            status = "Significant Growth"
        else:
            status = "Significant Decline"
    else:
        # If p >= 0.05, the trend is too noisy to be certain
        if slope > 0:
            status = "Slight Growth (Noisy)"
        else:
            status = "Slight Decline (Noisy)"
            
    # Define a "Stable" threshold (e.g. slope is very close to 0)
    if abs(slope) < 0.05: 
        status = "Stable / Flat"

    trend_results.append({
        'Topic_ID': topic_id,
        'Keywords': words_str,
        'Slope': slope,       # The rate of change
        'P_Value': p_value,   # The reliability score
        'R_Squared': r_value**2, # How well the line fits the data
        'Status': status,
        'Start_Pct': y_values[0],
        'End_Pct': y_values[-1]
    })
    
    # Print significant trends to console
    if p_value < 0.05:
        print(f"Topic {topic_id:2d}: Slope={slope:.3f} | p={p_value:.4f} | {status}")

# 4. Save and Sort Results
trend_df = pd.DataFrame(trend_results)

# Sort by Slope (Highest Growth to Deepest Decline)
trend_df = trend_df.sort_values(by='Slope', ascending=False)

output_file = os.path.join(OUTPUT_DIR, "04_statistical_trends.xlsx")
trend_df.to_excel(output_file, index=False)
print(f"\n✓ Analysis saved to: {output_file}")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: KEY FINDINGS")
print("="*80)

print(f"\n1. CULTURAL TERMS IN CORPUS:")
for term in CULTURAL_TERMS:
    n_topics = len(topics_with_terms[term])
    pct = (n_topics / len(original_representations)) * 100
    print(f"   '{term}': Present in {n_topics}/{len(original_representations)} topics ({pct:.1f}%)")

print(f"\n2. DOCUMENT DISTRIBUTION TRENDS (Statistical Slope):")
print("(Slope = Percentage points gained/lost per period)")

for term in CULTURAL_TERMS:
    # Get data for this term, sorted chronologically
    term_data = prevalence_df[prevalence_df['term'] == term].sort_values('period')
    
    if len(term_data) >= 2:
        # Prepare X (Time Steps: 0, 1, 2...) and Y (Percentages)
        x = range(len(term_data))
        y = term_data['percentage'].values
        
        # Calculate Linear Regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Interpret the result
        if p_value < 0.05:
            if slope > 0:
                trend_desc = "Significant Growth 📈"
            else:
                trend_desc = "Significant Decline 📉"
        else:
            trend_desc = "Stable / Noisy (Not Significant) 〰️"
            
        print(f"   '{term}': Slope {slope:+.2f} | p={p_value:.3f} | {trend_desc}")



print(f"\n3. TOPICS WITH CULTURAL VOCABULARY:")
cultural_topic_ids = sorted(cultural_topics)
print(f"   {len(cultural_topic_ids)} topics contain cultural terms: {cultural_topic_ids}")




print("\n" + "="*80)
print("ANALYSIS COMPLETE")
