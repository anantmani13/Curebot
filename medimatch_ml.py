"""
=============================================================================
MEDIMATCH ML ENGINE - Machine Learning Core
=============================================================================
This file contains ALL the Machine Learning logic for CureBot

Team: ML Team (2 members)
Purpose: Medicine recommendation using TF-IDF + Cosine Similarity

Technologies Used:
- TF-IDF Vectorizer (scikit-learn)
- Cosine Similarity
- N-grams (1-4)
- Symptom Synonym Expansion
- Disease Analytics

=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import plotly.graph_objects as go

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def extract_zip_if_needed(zip_path, extract_to='.'):
    """Extract ZIP file if it exists"""
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    return False


def load_data():
    """
    Load medicine datasets
    
    Returns:
        df1: Main medicine database (248,218 medicines)
        df2: Secondary medicine inventory (50,000 items)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load primary dataset
    file1 = os.path.join(script_dir, 'all_medicine databased.csv')
    if not os.path.exists(file1):
        print(f"ERROR: 'all_medicine databased.csv' not found in {script_dir}")
        return None, None
    
    df1 = pd.read_csv(file1, low_memory=False)

    # Load secondary dataset
    file2 = os.path.join(script_dir, 'medicine_dataset.csv')
    zip2 = os.path.join(script_dir, 'medicine_dataset.csv.zip')
    
    if not os.path.exists(file2):
        if os.path.exists(zip2):
            print("Unzipping dataset...")
            extract_zip_if_needed(zip2, script_dir)
    
    if not os.path.exists(file2):
        print(f"ERROR: 'medicine_dataset.csv' not found.")
        return None, None
        
    df2 = pd.read_csv(file2, low_memory=False)

    # Data Preprocessing
    df1, df2 = preprocess_data(df1, df2)
    
    return df1, df2


def preprocess_data(df1, df2):
    """
    Preprocess datasets for ML model
    
    Steps:
    1. Find all 'use' columns (use0, use1, use2, etc.)
    2. Combine all uses into single text field
    3. Add therapeutic class for better matching
    4. Clean medicine names
    """
    # Find use columns
    use_cols = [c for c in df1.columns if 'use' in c.lower()]
    
    # Combine uses into single text field
    df1['combined_use'] = df1[use_cols].fillna('').agg(' '.join, axis=1).str.lower()
    
    # Add therapeutic class for better semantic matching
    if 'Therapeutic Class' in df1.columns:
        df1['combined_use'] = df1['combined_use'] + ' ' + df1['Therapeutic Class'].fillna('').str.lower()
    
    # Clean medicine names
    if 'name' in df1.columns:
        df1['name_clean'] = df1['name'].str.lower().str.strip()
    
    if 'Name' in df2.columns:
        df2['Name_clean'] = df2['Name'].str.lower().str.strip()
    
    return df1, df2


# =============================================================================
# 2. TF-IDF MODEL TRAINING
# =============================================================================

def train_model(df1):
    """
    Train TF-IDF Vectorizer for medicine recommendation
    
    Model Parameters:
    - max_features: 15,000 (vocabulary size)
    - ngram_range: (1, 4) - unigrams to 4-grams
    - min_df: 2 (minimum document frequency)
    - max_df: 0.90 (maximum document frequency)
    - sublinear_tf: True (use log scaling)
    - smooth_idf: True (prevent zero division)
    - norm: 'l2' (normalize vectors)
    
    Returns:
        vectorizer: Trained TF-IDF Vectorizer
        tfidf_matrix: Sparse matrix of TF-IDF features
    """
    if df1 is None or df1.empty:
        return None, None
    
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=15000,           # Vocabulary size
        ngram_range=(1, 4),           # Up to 4-grams for phrases like "high blood pressure"
        min_df=2,                     # Minimum document frequency
        max_df=0.90,                  # Maximum document frequency
        sublinear_tf=True,            # Use log(1 + tf) scaling
        smooth_idf=True,              # Smooth IDF weights
        norm='l2'                     # L2 normalization
    )
    
    tfidf_matrix = vectorizer.fit_transform(df1['combined_use'].astype(str))
    
    print(f"✅ TF-IDF Model Trained:")
    print(f"   - Vocabulary Size: {len(vectorizer.vocabulary_)}")
    print(f"   - Matrix Shape: {tfidf_matrix.shape}")
    
    return vectorizer, tfidf_matrix


# =============================================================================
# 3. SYMPTOM SYNONYM EXPANSION
# =============================================================================

# 30 Symptom Categories with Medical Synonyms
SYMPTOM_SYNONYMS = {
    'headache': 'headache head pain migraine cephalalgia tension headache cluster headache sinus headache',
    'fever': 'fever pyrexia high temperature febrile hyperthermia chills',
    'cold': 'cold common cold flu influenza runny nose nasal congestion sneezing',
    'cough': 'cough dry cough wet cough productive cough bronchitis whooping',
    'pain': 'pain ache soreness discomfort body pain muscle pain joint pain arthralgia',
    'nausea': 'nausea vomiting upset stomach queasiness motion sickness antiemetic',
    'diarrhea': 'diarrhea loose motion stomach upset gastroenteritis dysentery',
    'allergy': 'allergy allergic reaction itching hives urticaria antihistamine rhinitis',
    'diabetes': 'diabetes blood sugar hyperglycemia glucose insulin antidiabetic',
    'hypertension': 'hypertension high blood pressure bp antihypertensive',
    'infection': 'infection bacterial viral fungal sepsis antibiotic antimicrobial',
    'anxiety': 'anxiety stress tension nervousness panic anxiolytic',
    'insomnia': 'insomnia sleeplessness sleep disorder trouble sleeping sedative hypnotic',
    'acidity': 'acidity heartburn acid reflux gastritis gerd antacid',
    'asthma': 'asthma wheezing breathing difficulty bronchospasm inhaler bronchodilator',
    'depression': 'depression sad mood antidepressant serotonin',
    'skin': 'skin rash eczema dermatitis psoriasis fungal cream ointment',
    'eye': 'eye vision conjunctivitis dry eye drops ophthalmic',
    'ear': 'ear pain otitis infection drops',
    'throat': 'throat sore pharyngitis tonsillitis strep',
    'vitamin': 'vitamin supplement deficiency multivitamin nutrition',
    'blood': 'blood anemia iron hemoglobin platelet',
    'heart': 'heart cardiac arrhythmia angina cardiovascular',
    'kidney': 'kidney renal urinary nephro',
    'liver': 'liver hepatic hepatitis cirrhosis',
    'thyroid': 'thyroid hypothyroid hyperthyroid levothyroxine',
    'cholesterol': 'cholesterol lipid statin triglyceride hdl ldl',
    'constipation': 'constipation laxative bowel movement stool softener',
    'dizziness': 'dizziness vertigo lightheaded balance',
    'fatigue': 'fatigue tiredness weakness energy exhaustion',
}


def expand_symptoms(user_input):
    """
    Expand user input with medical synonyms for better matching
    
    Example:
        Input: "headache"
        Output: "headache head pain migraine cephalalgia tension headache..."
    
    Args:
        user_input: User's symptom description
        
    Returns:
        Expanded symptom string with synonyms
    """
    expanded = user_input.lower()
    
    for symptom, synonyms in SYMPTOM_SYNONYMS.items():
        if symptom in expanded:
            expanded = expanded + ' ' + synonyms
    
    return expanded


# =============================================================================
# 4. MEDICINE SEARCH & RECOMMENDATION
# =============================================================================

def search_medicines(query, df1, vectorizer, tfidf_matrix, top_n=15):
    """
    Search medicines using TF-IDF + Cosine Similarity
    
    Algorithm:
    1. Expand query with symptom synonyms
    2. Transform query to TF-IDF vector
    3. Calculate cosine similarity with all medicines
    4. Return top N matches sorted by similarity score
    
    Args:
        query: User's symptom/medicine query
        df1: Medicine dataframe
        vectorizer: Trained TF-IDF vectorizer
        tfidf_matrix: Pre-computed TF-IDF matrix
        top_n: Number of results to return
        
    Returns:
        List of dictionaries with medicine details and match scores
    """
    if vectorizer is None or tfidf_matrix is None or df1 is None:
        return []
    
    # Step 1: Expand query with synonyms
    expanded_query = expand_symptoms(query)
    
    # Step 2: Transform to TF-IDF vector
    query_vector = vectorizer.transform([expanded_query])
    
    # Step 3: Calculate cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Step 4: Get top N indices
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Step 5: Build results
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.01:  # Minimum threshold
            medicine = df1.iloc[idx]
            
            # Calculate match percentage
            match_score = min(similarities[idx] * 100 * 1.5, 99.9)
            
            result = {
                'Medicine Name': medicine.get('name', medicine.get('Medicine Name', 'Unknown')),
                'Therapeutic Class': medicine.get('Therapeutic Class', 'General'),
                'Action Class': medicine.get('Action Class', 'N/A'),
                'Uses': get_medicine_uses(medicine),
                'Side Effects': get_side_effects(medicine),
                'Manufacturer': medicine.get('Manufacturer', 'N/A'),
                'Match Score': f"{match_score:.1f}%",
                'Raw Score': similarities[idx]
            }
            results.append(result)
    
    return results


def get_medicine_uses(medicine):
    """Extract all uses from medicine record"""
    uses = []
    for i in range(10):  # Check use0 to use9
        use_col = f'use{i}'
        if use_col in medicine and pd.notna(medicine[use_col]) and medicine[use_col]:
            uses.append(str(medicine[use_col]))
    return ', '.join(uses[:5]) if uses else 'General medicine'


def get_side_effects(medicine):
    """Extract side effects from medicine record"""
    effects = []
    for i in range(5):  # Check sideEffect0 to sideEffect4
        effect_col = f'sideEffect{i}'
        if effect_col in medicine and pd.notna(medicine[effect_col]) and medicine[effect_col]:
            effects.append(str(medicine[effect_col]))
    return ', '.join(effects[:3]) if effects else 'Consult doctor'


# =============================================================================
# 5. DISEASE ANALYTICS (AI-Generated Statistics)
# =============================================================================

DISEASE_STATS = {
    'headache': {'prevalence': 46, 'recovery_rate': 95, 'avg_duration': 2, 'severity': 'Low'},
    'fever': {'prevalence': 35, 'recovery_rate': 98, 'avg_duration': 3, 'severity': 'Moderate'},
    'cold': {'prevalence': 62, 'recovery_rate': 99, 'avg_duration': 7, 'severity': 'Low'},
    'cough': {'prevalence': 45, 'recovery_rate': 96, 'avg_duration': 10, 'severity': 'Low'},
    'pain': {'prevalence': 50, 'recovery_rate': 90, 'avg_duration': 5, 'severity': 'Moderate'},
    'diabetes': {'prevalence': 10, 'recovery_rate': 70, 'avg_duration': 365, 'severity': 'High'},
    'hypertension': {'prevalence': 25, 'recovery_rate': 75, 'avg_duration': 365, 'severity': 'High'},
    'allergy': {'prevalence': 30, 'recovery_rate': 85, 'avg_duration': 14, 'severity': 'Low'},
    'infection': {'prevalence': 40, 'recovery_rate': 92, 'avg_duration': 7, 'severity': 'Moderate'},
    'anxiety': {'prevalence': 18, 'recovery_rate': 80, 'avg_duration': 90, 'severity': 'Moderate'},
    'asthma': {'prevalence': 8, 'recovery_rate': 85, 'avg_duration': 365, 'severity': 'High'},
    'acidity': {'prevalence': 35, 'recovery_rate': 95, 'avg_duration': 7, 'severity': 'Low'},
    'insomnia': {'prevalence': 20, 'recovery_rate': 82, 'avg_duration': 30, 'severity': 'Moderate'},
    'diarrhea': {'prevalence': 28, 'recovery_rate': 97, 'avg_duration': 3, 'severity': 'Low'},
    'nausea': {'prevalence': 32, 'recovery_rate': 98, 'avg_duration': 2, 'severity': 'Low'},
    'default': {'prevalence': 25, 'recovery_rate': 88, 'avg_duration': 7, 'severity': 'Moderate'}
}


def get_disease_stats(symptom):
    """
    Get disease statistics for a symptom
    
    Args:
        symptom: User's symptom query
        
    Returns:
        Dictionary with prevalence, recovery_rate, avg_duration, severity
    """
    symptom_lower = symptom.lower()
    
    for key in DISEASE_STATS:
        if key in symptom_lower:
            return DISEASE_STATS[key]
    
    return DISEASE_STATS['default']


def create_analytics_graph(symptom):
    """
    Create interactive Plotly gauge charts for disease analytics
    
    Args:
        symptom: User's symptom query
        
    Returns:
        fig: Plotly figure with two gauges (Recovery Rate, Prevalence)
        stats: Dictionary with disease statistics
    """
    stats = get_disease_stats(symptom)
    
    fig = go.Figure()
    
    # Recovery Rate Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=stats['recovery_rate'],
        title={'text': "Recovery Rate %", 'font': {'size': 16, 'color': '#00695C'}},
        delta={'reference': 85, 'increasing': {'color': "#00695C"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#00695C"},
            'bar': {'color': "#00695C"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#B2DFDB",
            'steps': [
                {'range': [0, 50], 'color': '#FFCDD2'},
                {'range': [50, 75], 'color': '#FFE0B2'},
                {'range': [75, 100], 'color': '#C8E6C9'}
            ],
            'threshold': {
                'line': {'color': "#E53935", 'width': 4},
                'thickness': 0.75,
                'value': stats['recovery_rate']
            }
        },
        domain={'x': [0, 0.45], 'y': [0, 1]}
    ))
    
    # Prevalence Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=stats['prevalence'],
        title={'text': "Prevalence %", 'font': {'size': 16, 'color': '#00695C'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': "#4DB6AC"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#B2DFDB",
            'steps': [
                {'range': [0, 20], 'color': '#C8E6C9'},
                {'range': [20, 50], 'color': '#FFE0B2'},
                {'range': [50, 100], 'color': '#FFCDD2'}
            ]
        },
        domain={'x': [0.55, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(255,255,255,0.9)',
        font={'color': "#00695C", 'family': "Inter"},
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig, stats


# =============================================================================
# 6. GEMINI AI INTEGRATION
# =============================================================================

import urllib.request
import urllib.parse
import json

GEMINI_API_KEY = "AIzaSyDPT3BMbLEUj17haABGGpSsx70lDoPUgEA"


def get_gemini_health_advice(symptom, medicines):
    """
    Get AI health advice from Google Gemini
    
    Args:
        symptom: User's symptom
        medicines: List of recommended medicines
        
    Returns:
        AI-generated health advice string
    """
    try:
        medicine_list = ", ".join([m.get('Medicine Name', '') for m in medicines[:5]])
        
        prompt = f"""You are a helpful medical AI assistant. The user has symptoms: "{symptom}".
        Based on our database, we found these medicines: {medicine_list}.
        
        Please provide:
        1. Brief health tip (1-2 sentences)
        2. When to see a doctor
        3. Home remedies (if applicable)
        
        Keep response under 100 words. Be professional and caring."""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        
        data = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 150
            }
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers={
            'Content-Type': 'application/json'
        })
        
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode('utf-8'))
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Gemini API error: {e}")
    
    return None


# =============================================================================
# 7. MAIN - Test the ML Engine
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("   MEDIMATCH ML ENGINE - Testing")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading datasets...")
    df1, df2 = load_data()
    
    if df1 is not None:
        print(f"✅ Dataset 1 loaded: {len(df1):,} medicines")
        print(f"✅ Dataset 2 loaded: {len(df2):,} medicines")
        
        # Train model
        print("\n🧠 Training TF-IDF model...")
        vectorizer, tfidf_matrix = train_model(df1)
        
        # Test search
        print("\n🔍 Testing search for 'headache'...")
        results = search_medicines("headache", df1, vectorizer, tfidf_matrix, top_n=5)
        
        for i, med in enumerate(results, 1):
            print(f"\n   {i}. {med['Medicine Name']}")
            print(f"      Match: {med['Match Score']}")
            print(f"      Uses: {med['Uses'][:50]}...")
        
        # Test analytics
        print("\n📊 Testing disease analytics...")
        fig, stats = create_analytics_graph("headache")
        print(f"   Recovery Rate: {stats['recovery_rate']}%")
        print(f"   Prevalence: {stats['prevalence']}%")
        print(f"   Severity: {stats['severity']}")
        
        print("\n" + "=" * 60)
        print("   ✅ ML ENGINE TEST COMPLETE")
        print("=" * 60)
