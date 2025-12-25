"""
MEDIMATCH ML ENGINE - CureBot Machine Learning Core
TF-IDF + Cosine Similarity based medicine recommendation system
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import plotly.graph_objects as go
import urllib.request
import json

# DATA LOADING
def extract_zip_if_needed(zip_path, extract_to='.'):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    return False

def load_data():
    """Load medicine datasets (248K + 50K medicines)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    file1 = os.path.join(script_dir, 'all_medicine databased.csv')
    if not os.path.exists(file1):
        print(f"ERROR: 'all_medicine databased.csv' not found")
        return None, None
    
    df1 = pd.read_csv(file1, low_memory=False)

    file2 = os.path.join(script_dir, 'medicine_dataset.csv')
    zip2 = os.path.join(script_dir, 'medicine_dataset.csv.zip')
    
    if not os.path.exists(file2) and os.path.exists(zip2):
        extract_zip_if_needed(zip2, script_dir)
    
    if not os.path.exists(file2):
        print(f"ERROR: 'medicine_dataset.csv' not found")
        return None, None
        
    df2 = pd.read_csv(file2, low_memory=False)
    df1, df2 = preprocess_data(df1, df2)
    
    return df1, df2

def preprocess_data(df1, df2):
    """Preprocess: combine uses, add therapeutic class, clean names"""
    use_cols = [c for c in df1.columns if 'use' in c.lower()]
    df1['combined_use'] = df1[use_cols].fillna('').agg(' '.join, axis=1).str.lower()
    
    if 'Therapeutic Class' in df1.columns:
        df1['combined_use'] = df1['combined_use'] + ' ' + df1['Therapeutic Class'].fillna('').str.lower()
    
    if 'name' in df1.columns:
        df1['name_clean'] = df1['name'].str.lower().str.strip()
    if 'Name' in df2.columns:
        df2['Name_clean'] = df2['Name'].str.lower().str.strip()
    
    return df1, df2

# TF-IDF MODEL
def train_model(df1):
    """Train TF-IDF: 15K features, n-grams 1-4, sublinear TF, L2 norm"""
    if df1 is None or df1.empty:
        return None, None
    
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=15000,
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        smooth_idf=True,
        norm='l2'
    )
    
    tfidf_matrix = vectorizer.fit_transform(df1['combined_use'].astype(str))
    print(f"✅ TF-IDF Trained: {len(vectorizer.vocabulary_)} vocab, {tfidf_matrix.shape} matrix")
    
    return vectorizer, tfidf_matrix

# 30 SYMPTOM CATEGORIES WITH MEDICAL SYNONYMS
SYMPTOM_SYNONYMS = {
    'headache': 'headache head pain migraine cephalalgia tension headache cluster headache sinus headache',
    'fever': 'fever pyrexia high temperature febrile hyperthermia chills shivering',
    'cold': 'cold common cold flu influenza runny nose nasal congestion sneezing rhinitis',
    'cough': 'cough dry cough wet cough productive cough bronchitis whooping tussis',
    'pain': 'pain ache soreness discomfort body pain muscle pain joint pain arthralgia myalgia',
    'nausea': 'nausea vomiting upset stomach queasiness motion sickness antiemetic',
    'diarrhea': 'diarrhea loose motion stomach upset gastroenteritis dysentery',
    'allergy': 'allergy allergic reaction itching hives urticaria antihistamine rhinitis',
    'diabetes': 'diabetes blood sugar hyperglycemia glucose insulin antidiabetic',
    'hypertension': 'hypertension high blood pressure bp antihypertensive cardiovascular',
    'infection': 'infection bacterial viral fungal sepsis antibiotic antimicrobial',
    'anxiety': 'anxiety stress tension nervousness panic anxiolytic restless',
    'insomnia': 'insomnia sleeplessness sleep disorder trouble sleeping sedative hypnotic',
    'acidity': 'acidity heartburn acid reflux gastritis gerd antacid dyspepsia',
    'asthma': 'asthma wheezing breathing difficulty bronchospasm inhaler bronchodilator',
    'depression': 'depression sad mood antidepressant serotonin melancholy',
    'skin': 'skin rash eczema dermatitis psoriasis fungal cream ointment topical',
    'eye': 'eye vision conjunctivitis dry eye drops ophthalmic ocular',
    'ear': 'ear pain otitis infection drops otic hearing tinnitus',
    'throat': 'throat sore pharyngitis tonsillitis strep laryngitis',
    'vitamin': 'vitamin supplement deficiency multivitamin nutrition minerals',
    'blood': 'blood anemia iron hemoglobin platelet hematologic',
    'heart': 'heart cardiac arrhythmia angina cardiovascular palpitation',
    'kidney': 'kidney renal urinary nephro bladder',
    'liver': 'liver hepatic hepatitis cirrhosis hepato biliary',
    'thyroid': 'thyroid hypothyroid hyperthyroid levothyroxine goiter',
    'cholesterol': 'cholesterol lipid statin triglyceride hdl ldl',
    'constipation': 'constipation laxative bowel movement stool softener',
    'dizziness': 'dizziness vertigo lightheaded balance spinning faint',
    'fatigue': 'fatigue tiredness weakness energy exhaustion lethargy',
}

def expand_symptoms(user_input):
    """Expand user input with medical synonyms for better matching"""
    expanded = user_input.lower()
    for symptom, synonyms in SYMPTOM_SYNONYMS.items():
        if symptom in expanded:
            expanded = expanded + ' ' + synonyms
    return expanded

# CORE SEARCH ALGORITHM
def search_medicines(query, df1, vectorizer, tfidf_matrix, top_n=15):
    """TF-IDF + Cosine Similarity search with synonym expansion"""
    if vectorizer is None or tfidf_matrix is None or df1 is None:
        return []
    
    expanded_query = expand_symptoms(query)
    query_vector = vectorizer.transform([expanded_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.01:
            medicine = df1.iloc[idx]
            match_score = min(similarities[idx] * 100 * 1.5, 99.9)
            
            results.append({
                'Medicine Name': medicine.get('name', medicine.get('Medicine Name', 'Unknown')),
                'Therapeutic Class': medicine.get('Therapeutic Class', 'General'),
                'Action Class': medicine.get('Action Class', 'N/A'),
                'Uses': get_medicine_uses(medicine),
                'Side Effects': get_side_effects(medicine),
                'Manufacturer': medicine.get('Manufacturer', 'N/A'),
                'Match Score': f"{match_score:.1f}%",
                'Raw Score': similarities[idx]
            })
    
    return results

def get_medicine_uses(medicine):
    uses = []
    for i in range(10):
        use_col = f'use{i}'
        if use_col in medicine and pd.notna(medicine[use_col]) and medicine[use_col]:
            uses.append(str(medicine[use_col]))
    return ', '.join(uses[:5]) if uses else 'General medicine'

def get_side_effects(medicine):
    effects = []
    for i in range(5):
        effect_col = f'sideEffect{i}'
        if effect_col in medicine and pd.notna(medicine[effect_col]) and medicine[effect_col]:
            effects.append(str(medicine[effect_col]))
    return ', '.join(effects[:3]) if effects else 'Consult doctor'

# DISEASE ANALYTICS
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
    'skin': {'prevalence': 22, 'recovery_rate': 88, 'avg_duration': 14, 'severity': 'Low'},
    'eye': {'prevalence': 15, 'recovery_rate': 92, 'avg_duration': 7, 'severity': 'Low'},
    'ear': {'prevalence': 12, 'recovery_rate': 94, 'avg_duration': 7, 'severity': 'Low'},
    'throat': {'prevalence': 38, 'recovery_rate': 96, 'avg_duration': 5, 'severity': 'Low'},
    'heart': {'prevalence': 8, 'recovery_rate': 78, 'avg_duration': 365, 'severity': 'High'},
    'kidney': {'prevalence': 5, 'recovery_rate': 75, 'avg_duration': 180, 'severity': 'High'},
    'liver': {'prevalence': 4, 'recovery_rate': 72, 'avg_duration': 180, 'severity': 'High'},
    'thyroid': {'prevalence': 6, 'recovery_rate': 85, 'avg_duration': 365, 'severity': 'Moderate'},
    'cholesterol': {'prevalence': 20, 'recovery_rate': 80, 'avg_duration': 365, 'severity': 'Moderate'},
    'constipation': {'prevalence': 25, 'recovery_rate': 95, 'avg_duration': 5, 'severity': 'Low'},
    'dizziness': {'prevalence': 18, 'recovery_rate': 90, 'avg_duration': 3, 'severity': 'Low'},
    'fatigue': {'prevalence': 35, 'recovery_rate': 85, 'avg_duration': 14, 'severity': 'Low'},
    'default': {'prevalence': 25, 'recovery_rate': 88, 'avg_duration': 7, 'severity': 'Moderate'}
}

def get_disease_stats(symptom):
    symptom_lower = symptom.lower()
    for key in DISEASE_STATS:
        if key in symptom_lower:
            return DISEASE_STATS[key]
    return DISEASE_STATS['default']

def create_analytics_graph(symptom):
    """Create Plotly gauge charts for disease analytics"""
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
            'threshold': {'line': {'color': "#E53935", 'width': 4}, 'thickness': 0.75, 'value': stats['recovery_rate']}
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

# GEMINI AI INTEGRATION
GEMINI_API_KEY = "AIzaSyDPT3BMbLEUj17haABGGpSsx70lDoPUgEA"

def get_gemini_health_advice(symptom, medicines):
    """Get AI health advice from Google Gemini API"""
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
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 150}
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Gemini API error: {e}")
    
    return None

# GLOBAL MODEL STATE
df1, df2 = None, None
vectorizer, tfidf_matrix = None, None
DATA_LOADED = False

def initialize_model():
    """Initialize ML model at startup"""
    global df1, df2, vectorizer, tfidf_matrix, DATA_LOADED
    
    print("=" * 50)
    print("   MEDIMATCH ML ENGINE - Initializing")
    print("=" * 50)
    
    df1, df2 = load_data()
    
    if df1 is not None:
        print(f"✅ Dataset 1: {len(df1):,} medicines")
        if df2 is not None:
            print(f"✅ Dataset 2: {len(df2):,} medicines")
        
        vectorizer, tfidf_matrix = train_model(df1)
        DATA_LOADED = True
        print("✅ ML Engine Ready!")
    else:
        DATA_LOADED = False
        print("❌ Failed to load datasets")
    
    print("=" * 50)
    return DATA_LOADED

def get_recommendations(user_input, top_n=15):
    """High-level API to get medicine recommendations"""
    if not DATA_LOADED:
        return []
    return search_medicines(user_input, df1, vectorizer, tfidf_matrix, top_n)

# TEST
if __name__ == '__main__':
    success = initialize_model()
    
    if success:
        print("\n🔍 Testing 'headache'...")
        results = get_recommendations("headache", top_n=5)
        
        for i, med in enumerate(results, 1):
            print(f"   {i}. {med['Medicine Name']} - {med['Match Score']}")
        
        print("\n📊 Analytics...")
        fig, stats = create_analytics_graph("headache")
        print(f"   Recovery: {stats['recovery_rate']}%, Severity: {stats['severity']}")
        
        print("\n✅ ML ENGINE TEST COMPLETE")
