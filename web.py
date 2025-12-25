import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import difflib
import zipfile
import os
import re
import json
import hashlib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Semantic Search with Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
    print("✅ Sentence Transformers loaded successfully")
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠️ Sentence Transformers not available, using TF-IDF only")

# =============================================================================
# APP BRANDING & CONFIGURATION
# =============================================================================
APP_NAME = "CureBot"
APP_TAGLINE = "Your AI-Powered Medicine Assistant"
APP_VERSION = "3.0"

# Hospital Green Theme Colors
PRIMARY_GREEN = "#00695C"      # Deep Hospital Green
LIGHT_GREEN = "#4DB6AC"        # Light Teal
ACCENT_GREEN = "#B2DFDB"       # Very Light Green
MEDICAL_RED = "#D32F2F"        # Medical Cross Red
BG_COLOR = "#E0F2F1"           # Soft Green Background
EMERGENCY_RED = "#B71C1C"      # Emergency Dark Red

# =============================================================================
# GOOGLE API CONFIGURATION
# =============================================================================
GOOGLE_CLIENT_ID = "1079027064414-82gdpim62um96jjgg91tcct8oucapphk.apps.googleusercontent.com"
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"  # Optional - OpenStreetMap is used instead (FREE)

# Google Gemini API (Get free key from https://aistudio.google.com/)
GEMINI_API_KEY = "AIzaSyDPT3BMbLEUj17haABGGpSsx70lDoPUgEA"

# Check if APIs are configured
GOOGLE_AUTH_ENABLED = GOOGLE_CLIENT_ID != "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com"
GOOGLE_MAPS_ENABLED = GOOGLE_MAPS_API_KEY != "YOUR_GOOGLE_MAPS_API_KEY"
GEMINI_ENABLED = GEMINI_API_KEY != "YOUR_GEMINI_API_KEY"

# =============================================================================
# GEMINI AI INTEGRATION (Google AI Studio)
# =============================================================================
import urllib.request
import urllib.parse

def get_gemini_health_advice(symptom, medicines):
    """Get AI health advice from Google Gemini"""
    if not GEMINI_ENABLED:
        return None
    
    try:
        # Prepare the prompt
        medicine_list = ", ".join([m.get('Medicine Name', '') for m in medicines[:5]])
        
        prompt = f"""You are a helpful medical AI assistant. The user has symptoms: "{symptom}".
        Based on our database, we found these medicines: {medicine_list}.
        
        Please provide:
        1. Brief health tip (1-2 sentences)
        2. When to see a doctor
        3. Home remedies (if applicable)
        
        Keep response under 100 words. Be professional and caring."""
        
        # Gemini API request (using gemini-2.0-flash model)
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
# DISEASE ANALYTICS DATA (AI-Generated Statistics)
# =============================================================================
DISEASE_STATS = {
    'headache': {'prevalence': 46, 'recovery_rate': 95, 'avg_duration': 2, 'severity': 'Low'},
    'fever': {'prevalence': 38, 'recovery_rate': 98, 'avg_duration': 3, 'severity': 'Medium'},
    'cold': {'prevalence': 62, 'recovery_rate': 99, 'avg_duration': 7, 'severity': 'Low'},
    'cough': {'prevalence': 55, 'recovery_rate': 97, 'avg_duration': 10, 'severity': 'Low'},
    'diabetes': {'prevalence': 10, 'recovery_rate': 85, 'avg_duration': 365, 'severity': 'High'},
    'hypertension': {'prevalence': 26, 'recovery_rate': 80, 'avg_duration': 365, 'severity': 'High'},
    'asthma': {'prevalence': 8, 'recovery_rate': 90, 'avg_duration': 365, 'severity': 'Medium'},
    'anxiety': {'prevalence': 18, 'recovery_rate': 75, 'avg_duration': 90, 'severity': 'Medium'},
    'depression': {'prevalence': 5, 'recovery_rate': 70, 'avg_duration': 180, 'severity': 'High'},
    'allergy': {'prevalence': 30, 'recovery_rate': 92, 'avg_duration': 14, 'severity': 'Low'},
    'skin': {'prevalence': 20, 'recovery_rate': 88, 'avg_duration': 21, 'severity': 'Low'},
    'acidity': {'prevalence': 25, 'recovery_rate': 94, 'avg_duration': 7, 'severity': 'Low'},
    'pain': {'prevalence': 40, 'recovery_rate': 96, 'avg_duration': 5, 'severity': 'Medium'},
    'insomnia': {'prevalence': 15, 'recovery_rate': 78, 'avg_duration': 30, 'severity': 'Medium'},
}

def create_disease_analytics_graph(symptom):
    """Create interactive Plotly graphs for disease analytics"""
    symptom_lower = symptom.lower()
    
    # Find matching disease
    matched_disease = None
    for disease in DISEASE_STATS:
        if disease in symptom_lower or symptom_lower in disease:
            matched_disease = disease
            break
    
    if not matched_disease:
        matched_disease = 'cold'  # Default
    
    stats = DISEASE_STATS[matched_disease]
    
    # Create gauge chart for recovery rate
    fig = go.Figure()
    
    # Recovery Rate Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=stats['recovery_rate'],
        title={'text': f"Recovery Rate", 'font': {'size': 16, 'color': '#00695C'}},
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
# 1. ML BACKEND - Data Loading & Model Training
# =============================================================================

def extract_zip_if_needed(zip_path, extract_to='.'):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    return False

def load_data():
    """Load both medicine datasets"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    file1 = os.path.join(script_dir, 'all_medicine databased.csv')
    if not os.path.exists(file1):
        print(f"CRITICAL ERROR: 'all_medicine databased.csv' not found in {script_dir}")
        return None, None
    
    df1 = pd.read_csv(file1, low_memory=False)

    file2 = os.path.join(script_dir, 'medicine_dataset.csv')
    zip2 = os.path.join(script_dir, 'medicine_dataset.csv.zip')
    
    if not os.path.exists(file2):
        if os.path.exists(zip2):
            print("Unzipping dataset...")
            extract_zip_if_needed(zip2, script_dir)
    
    if not os.path.exists(file2):
        print(f"CRITICAL ERROR: 'medicine_dataset.csv' not found.")
        return None, None
        
    df2 = pd.read_csv(file2, low_memory=False)

    # Enhanced Preprocessing
    use_cols = [c for c in df1.columns if 'use' in c.lower()]
    side_effect_cols = [c for c in df1.columns if 'sideEffect' in c]
    
    # Combine uses and add therapeutic class for better matching
    df1['combined_use'] = df1[use_cols].fillna('').agg(' '.join, axis=1).str.lower()
    
    # Add therapeutic class to combined text for better semantic matching
    if 'Therapeutic Class' in df1.columns:
        df1['combined_use'] = df1['combined_use'] + ' ' + df1['Therapeutic Class'].fillna('').str.lower()
    
    if 'name' in df1.columns:
        df1['name_clean'] = df1['name'].str.lower().str.strip()
    
    if 'Name' in df2.columns:
        df2['Name_clean'] = df2['Name'].str.lower().str.strip()
    
    return df1, df2

def train_model(df1):
    """Train Enhanced TF-IDF model with advanced features"""
    if df1 is None or df1.empty:
        return None, None
    
    # Advanced vectorizer with better parameters
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=15000,           # More features
        ngram_range=(1, 4),           # Up to 4-grams for phrases
        min_df=2,                     
        max_df=0.90,                  
        sublinear_tf=True,
        smooth_idf=True,
        norm='l2'
    )
    tfidf_matrix = vectorizer.fit_transform(df1['combined_use'].astype(str))
    return vectorizer, tfidf_matrix

# SEMANTIC SEARCH - Sentence Transformers
semantic_model = None
medicine_embeddings = None

def load_semantic_model():
    """Load lightweight semantic search model"""
    global semantic_model
    if not SEMANTIC_AVAILABLE:
        return None
    try:
        # Using all-MiniLM-L6-v2 - fast & accurate (22M params, 80MB)
        print("🔄 Loading semantic search model...")
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Semantic model loaded: all-MiniLM-L6-v2")
        return semantic_model
    except Exception as e:
        print(f"⚠️ Semantic model failed: {e}")
        return None

def create_medicine_embeddings(df1, sample_size=5000):
    """Create embeddings for medicine uses (sampled for speed)"""
    global medicine_embeddings
    if semantic_model is None or df1 is None:
        return None
    
    try:
        print(f"🔄 Creating medicine embeddings (sampling {sample_size} of {len(df1)})...")
        
        # Sample for faster processing
        if len(df1) > sample_size:
            sample_indices = np.random.choice(len(df1), sample_size, replace=False)
            texts = df1.iloc[sample_indices]['combined_use'].astype(str).tolist()
        else:
            sample_indices = np.arange(len(df1))
            texts = df1['combined_use'].astype(str).tolist()
        
        # Batch encode for speed
        embeddings = semantic_model.encode(texts, batch_size=64, show_progress_bar=False)
        medicine_embeddings = {'embeddings': embeddings, 'indices': sample_indices}
        
        print(f"✅ Created {len(embeddings)} medicine embeddings")
        return medicine_embeddings
    except Exception as e:
        print(f"⚠️ Embedding creation failed: {e}")
        return None

def semantic_search(query, df1, top_n=50):
    """Semantic search using sentence embeddings"""
    if semantic_model is None or medicine_embeddings is None:
        return pd.DataFrame()
    
    try:
        # Encode query
        query_embedding = semantic_model.encode([query.lower()])
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, medicine_embeddings['embeddings']).flatten()
        
        # Get top matches
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Map back to original dataframe indices
        original_indices = medicine_embeddings['indices'][top_indices]
        scores = similarities[top_indices]
        
        # Filter by threshold
        valid = scores > 0.25
        return df1.iloc[original_indices[valid]]
    except Exception as e:
        print(f"Semantic search error: {e}")
        return pd.DataFrame()

# Enhanced Symptom Synonyms for smarter matching
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
    """Expand user input with synonyms for better matching"""
    expanded = user_input.lower()
    for key, synonyms in SYMPTOM_SYNONYMS.items():
        if key in expanded:
            expanded = expanded + ' ' + synonyms
    return expanded

# SMART QUERY VALIDATION - Reject non-medical queries
MEDICAL_KEYWORDS = {
    'pain', 'ache', 'fever', 'cold', 'cough', 'headache', 'stomach', 'nausea', 'vomiting',
    'diarrhea', 'allergy', 'rash', 'infection', 'diabetes', 'sugar', 'blood', 'pressure',
    'heart', 'chest', 'breathing', 'asthma', 'anxiety', 'stress', 'depression', 'sleep',
    'insomnia', 'tired', 'fatigue', 'weakness', 'dizziness', 'vertigo', 'eye', 'ear',
    'throat', 'skin', 'joint', 'muscle', 'back', 'knee', 'leg', 'arm', 'neck', 'shoulder',
    'vitamin', 'supplement', 'tablet', 'medicine', 'drug', 'capsule', 'syrup', 'injection',
    'antibiotic', 'painkiller', 'treatment', 'cure', 'remedy', 'health', 'medical', 'doctor',
    'hospital', 'disease', 'illness', 'symptom', 'sickness', 'flu', 'viral', 'bacterial',
    'fungal', 'wound', 'injury', 'burn', 'cut', 'swelling', 'inflammation', 'cramp',
    'migraine', 'acidity', 'gas', 'bloating', 'constipation', 'digestion', 'liver', 'kidney',
    'thyroid', 'cholesterol', 'weight', 'obesity', 'pregnancy', 'periods', 'menstrual',
    'bone', 'fracture', 'sprain', 'arthritis', 'cancer', 'tumor', 'ulcer', 'hernia',
    'piles', 'hemorrhoids', 'urinary', 'prostate', 'sexual', 'hormonal', 'immunity',
    'covid', 'corona', 'malaria', 'dengue', 'typhoid', 'jaundice', 'hepatitis', 'tb',
    'tuberculosis', 'hiv', 'aids', 'epilepsy', 'seizure', 'paralysis', 'stroke', 'bp',
    'sugar', 'insulin', 'glucose', 'hemoglobin', 'platelet', 'wbc', 'rbc', 'uric',
    'creatinine', 'bilirubin', 'sgpt', 'sgot', 'ecg', 'xray', 'scan', 'mri', 'ct',
    'paracetamol', 'ibuprofen', 'aspirin', 'crocin', 'dolo', 'combiflam', 'calpol',
    'azithromycin', 'amoxicillin', 'cetirizine', 'montair', 'pantoprazole', 'omeprazole',
    'metformin', 'amlodipine', 'atorvastatin', 'losartan', 'telmisartan', 'vitamin',
    'zinc', 'iron', 'calcium', 'b12', 'd3', 'folic', 'biotin', 'omega', 'protein',
    # Hindi/common terms
    'dard', 'bukhar', 'khansi', 'zukam', 'sir', 'pet', 'kamar', 'ghutna', 'gala',
    'aankh', 'kaan', 'dant', 'tooth', 'dental', 'oral', 'mouth', 'gum', 'tongue'
}

NON_MEDICAL_PATTERNS = [
    'hi', 'hello', 'hey', 'bye', 'thanks', 'thank', 'ok', 'okay', 'yes', 'no', 'please',
    'help', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would',
    'padhai', 'study', 'exam', 'school', 'college', 'job', 'work', 'money', 'salary',
    'weather', 'time', 'date', 'day', 'movie', 'song', 'music', 'game', 'play', 'food',
    'recipe', 'cook', 'travel', 'hotel', 'flight', 'train', 'bus', 'car', 'bike',
    'phone', 'laptop', 'computer', 'internet', 'wifi', 'app', 'website', 'download',
    'love', 'relationship', 'friend', 'family', 'marriage', 'wedding', 'party',
    'news', 'politics', 'sports', 'cricket', 'football', 'ipl', 'match',
    'good', 'bad', 'nice', 'great', 'awesome', 'cool', 'hot', 'beautiful',
    'kaise', 'kya', 'kab', 'kahan', 'kaun', 'kyun', 'accha', 'theek', 'sahi',
    'padhai nhi', 'bore', 'boring', 'lonely', 'sad', 'happy', 'angry', 'hungry',
    'thirsty', 'sleepy', 'lazy', 'busy', 'free', 'available'
]

def is_medical_query(user_input):
    """Check if query is medical-related"""
    query = user_input.lower().strip()
    
    # Too short queries
    if len(query) < 3:
        return False, "Please describe your health concern in more detail."
    
    # Check for greetings/non-medical patterns
    query_words = query.split()
    if len(query_words) <= 2:
        for pattern in NON_MEDICAL_PATTERNS:
            if pattern in query:
                return False, "I'm CureBot, a medicine recommendation assistant. Please describe your health symptoms or medical concerns, and I'll help find suitable medicines."
    
    # Check for medical keywords
    has_medical_keyword = False
    for keyword in MEDICAL_KEYWORDS:
        if keyword in query:
            has_medical_keyword = True
            break
    
    if not has_medical_keyword and len(query_words) <= 3:
        return False, "I can only help with health and medicine related queries. Please describe your symptoms like 'headache', 'fever', 'stomach pain', etc."
    
    return True, None

# --- Initialize Data Globally ---
print(f"🚀 Initializing {APP_NAME} v{APP_VERSION}...")
df1, df2 = load_data()

if df1 is not None and df2 is not None:
    vectorizer, tfidf_matrix = train_model(df1)
    DATA_LOADED = True
    print("✅ Data Loaded & Enhanced ML Model Trained Successfully!")
    print(f"   - Dataset 1: {len(df1):,} medicines")
    print(f"   - Dataset 2: {len(df2):,} inventory items")
    print(f"   - Model: Advanced TF-IDF (n-grams 1-4) + {len(SYMPTOM_SYNONYMS)} symptom categories")
    
    # Load Semantic Search Model
    if SEMANTIC_AVAILABLE:
        load_semantic_model()
        if semantic_model is not None:
            create_medicine_embeddings(df1, sample_size=8000)
            print("   - Semantic Search: ✅ Enabled (all-MiniLM-L6-v2)")
        else:
            print("   - Semantic Search: ❌ Disabled")
    else:
        print("   - Semantic Search: ❌ Not installed")
else:
    vectorizer, tfidf_matrix = None, None
    DATA_LOADED = False
    print("❌ WARNING: App starting in 'No Data' mode.")

# =============================================================================
# 2. ML CORE FUNCTIONS - Enhanced Recommendation Engine
# =============================================================================

def get_recommendations(user_input):
    """Hybrid recommendation: TF-IDF + Semantic Search"""
    if not DATA_LOADED:
        return pd.DataFrame()
    
    # Expand input with synonyms
    expanded_input = expand_symptoms(user_input)
    
    # 1. TF-IDF Search
    user_vec = vectorizer.transform([expanded_input])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    tfidf_indices = cosine_sim.argsort()[-100:][::-1]
    tfidf_scores = {i: cosine_sim[i] for i in tfidf_indices if cosine_sim[i] > 0.02}
    
    # 2. Semantic Search (if available)
    semantic_indices = set()
    if semantic_model is not None and medicine_embeddings is not None:
        try:
            query_embedding = semantic_model.encode([user_input.lower()])
            similarities = cosine_similarity(query_embedding, medicine_embeddings['embeddings']).flatten()
            top_sem_indices = similarities.argsort()[-50:][::-1]
            
            for idx in top_sem_indices:
                if similarities[idx] > 0.3:
                    original_idx = medicine_embeddings['indices'][idx]
                    semantic_indices.add(original_idx)
                    # Boost score if also in TF-IDF
                    if original_idx in tfidf_scores:
                        tfidf_scores[original_idx] *= 1.5  # Boost hybrid matches
                    else:
                        tfidf_scores[original_idx] = similarities[idx] * 0.8
        except Exception as e:
            print(f"Semantic search fallback: {e}")
    
    # 3. Combine and rank
    all_indices = sorted(tfidf_scores.keys(), key=lambda x: tfidf_scores[x], reverse=True)
    
    # Dynamic threshold
    word_count = len(user_input.split())
    threshold = 0.03 if word_count <= 1 else 0.05 if word_count <= 3 else 0.07
    
    relevant_indices = [i for i in all_indices if tfidf_scores[i] > threshold][:150]
    
    # Fallback
    if not relevant_indices and all_indices:
        relevant_indices = all_indices[:25]
    
    return df1.iloc[relevant_indices]

def get_medicine_details(candidates_df1, df2):
    """Get medicine details with enriched information"""
    if candidates_df1.empty:
        return []

    valid_medicines = []
    df2_names = set(df2['Name_clean'].unique()) if df2 is not None else set()
    
    for idx, row in candidates_df1.iterrows():
        if 'name' not in row: 
            continue
        
        medicine_name = str(row['name'])
        first_word = medicine_name.split()[0].lower()
        
        therapeutic_class = row.get('Therapeutic Class', 'General')
        chemical_class = row.get('Chemical Class', 'N/A')
        habit_forming = row.get('Habit Forming', 'No')
        
        primary_use = str(row.get('use0', 'General use'))
        if primary_use == 'nan':
            primary_use = 'General use'
        
        # Get side effects
        side_effects = []
        for i in range(3):
            se = row.get(f'sideEffect{i}', '')
            if pd.notna(se) and str(se) != 'nan':
                side_effects.append(str(se))
        
        form = 'Tablet'
        classification = 'Prescription'
        
        matches = difflib.get_close_matches(first_word, df2_names, n=1, cutoff=0.5)
        if matches:
            match_row = df2[df2['Name_clean'] == matches[0]].iloc[0]
            form = match_row.get('Dosage Form', form)
            classification = match_row.get('Classification', classification)
        
        valid_medicines.append({
            'Medicine Name': medicine_name.title(),
            'Primary Use': primary_use.replace('Treatment of ', '').title(),
            'Form': form,
            'Class': str(therapeutic_class) if str(therapeutic_class) != 'nan' else 'General',
            'Type': classification,
            'Side Effects': ', '.join(side_effects[:2]) if side_effects else 'Consult doctor'
        })
    
    # Remove duplicates and limit results
    seen = set()
    unique_medicines = []
    for med in valid_medicines:
        med_key = med['Medicine Name'].lower()
        if med_key not in seen:
            seen.add(med_key)
            unique_medicines.append(med)
            if len(unique_medicines) >= 12:
                break
    
    return unique_medicines

def get_ai_recommendation(user_input):
    """Main ML function with smart query validation"""
    if not DATA_LOADED:
        return [], None
    
    # Smart validation - reject non-medical queries
    is_valid, error_msg = is_medical_query(user_input)
    if not is_valid:
        return [], error_msg
    
    candidates = get_recommendations(user_input)
    
    if candidates.empty:
        return [], "No medicines found for your query. Try describing your symptoms differently."
    
    valid_medicines = get_medicine_details(candidates, df2)
    
    if not valid_medicines:
        return [], "No matching medicines found. Please describe your symptoms more specifically."
    
    return valid_medicines, None

# =============================================================================
# 3. DASH APP SETUP
# =============================================================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # For deployment

# =============================================================================
# 4. CSS STYLING - Premium Hospital Theme
# =============================================================================

maps_script = f'<script src="https://maps.googleapis.com/maps/api/js?key={GOOGLE_MAPS_API_KEY}&libraries=places"></script>' if GOOGLE_MAPS_ENABLED else ''

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{APP_NAME} - {APP_TAGLINE}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">
        <!-- Leaflet.js - FREE OpenStreetMap -->
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
        <!-- Google Sign-In -->
        <script src="https://accounts.google.com/gsi/client" async defer></script>
        {{%css%}}
        <style>
            :root {{
                --primary: #00695C;
                --primary-light: #4DB6AC;
                --primary-dark: #004D40;
                --accent: #B2DFDB;
                --medical-red: #E53935;
                --bg-gradient: linear-gradient(135deg, #E0F2F1 0%, #B2DFDB 50%, #80CBC4 100%);
                --shadow-soft: 0 4px 20px rgba(0,105,92,0.15);
                --shadow-medium: 0 8px 30px rgba(0,105,92,0.2);
            }}
            
            * {{ 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                box-sizing: border-box;
            }}
            
            body {{ 
                background: var(--bg-gradient);
                margin: 0;
                min-height: 100vh;
            }}
            
            /* ==================== LOGIN PAGE STYLES ==================== */
            .login-page {{
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                background: linear-gradient(135deg, #004D40 0%, #00695C 30%, #00897B 70%, #4DB6AC 100%);
                padding: 20px;
            }}
            .login-container {{
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(20px);
                border-radius: 32px;
                padding: 50px 45px;
                max-width: 420px;
                width: 100%;
                box-shadow: 0 25px 80px rgba(0,0,0,0.3);
                text-align: center;
                animation: loginSlideIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            }}
            @keyframes loginSlideIn {{
                from {{ opacity: 0; transform: translateY(40px) scale(0.95); }}
                to {{ opacity: 1; transform: translateY(0) scale(1); }}
            }}
            .login-logo {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 90px;
                height: 90px;
                background: linear-gradient(135deg, #E53935 0%, #C62828 100%);
                border-radius: 24px;
                margin-bottom: 25px;
                box-shadow: 0 8px 30px rgba(229,57,53,0.4);
                animation: logoPulse 2s ease-in-out infinite;
                position: relative;
            }}
            .login-logo::before,
            .login-logo::after {{
                content: '';
                position: absolute;
                background: white;
                border-radius: 5px;
            }}
            .login-logo::before {{
                width: 45px;
                height: 15px;
            }}
            .login-logo::after {{
                width: 15px;
                height: 45px;
            }}
            @keyframes logoPulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
            }}
            .login-title {{
                font-size: 2.5rem;
                font-weight: 800;
                color: #00695C;
                margin: 0 0 8px 0;
                letter-spacing: 2px;
            }}
            .login-subtitle {{
                color: #4DB6AC;
                font-size: 1rem;
                margin-bottom: 35px;
                font-weight: 500;
            }}
            .login-divider {{
                display: flex;
                align-items: center;
                margin: 30px 0;
                color: #999;
                font-size: 0.85rem;
            }}
            .login-divider::before,
            .login-divider::after {{
                content: '';
                flex: 1;
                height: 1px;
                background: linear-gradient(90deg, transparent, #ccc, transparent);
            }}
            .login-divider span {{
                padding: 0 15px;
            }}
            .google-btn-container {{
                display: flex;
                justify-content: center;
                margin: 25px 0;
            }}
            .skip-login {{
                background: transparent;
                border: 2px solid #4DB6AC;
                color: #00695C;
                padding: 14px 35px;
                border-radius: 30px;
                font-size: 0.95rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 15px;
            }}
            .skip-login:hover {{
                background: #00695C;
                color: white;
                border-color: #00695C;
            }}
            .login-features {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 35px;
                padding-top: 25px;
                border-top: 1px solid #eee;
            }}
            .login-feature {{
                text-align: center;
                color: #666;
            }}
            .login-feature-icon {{
                font-size: 1.5rem;
                margin-bottom: 5px;
            }}
            .login-feature-text {{
                font-size: 0.75rem;
                font-weight: 500;
            }}
            .user-info {{
                display: flex;
                align-items: center;
                gap: 12px;
                background: rgba(0,105,92,0.1);
                padding: 10px 18px;
                border-radius: 30px;
                margin-left: 20px;
            }}
            .user-avatar {{
                width: 35px;
                height: 35px;
                border-radius: 50%;
                border: 2px solid white;
            }}
            .user-name {{
                color: white;
                font-weight: 600;
                font-size: 0.9rem;
            }}
            .logout-btn {{
                background: rgba(255,255,255,0.2);
                border: none;
                color: white;
                padding: 6px 12px;
                border-radius: 15px;
                cursor: pointer;
                font-size: 0.8rem;
                margin-left: 8px;
                transition: all 0.3s;
            }}
            .logout-btn:hover {{
                background: rgba(255,255,255,0.3);
            }}
            
            /* Premium Scrollbar */
            ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
            ::-webkit-scrollbar-track {{ background: rgba(0,105,92,0.1); border-radius: 10px; }}
            ::-webkit-scrollbar-thumb {{ 
                background: linear-gradient(180deg, var(--primary), var(--primary-light)); 
                border-radius: 10px;
                border: 2px solid transparent;
                background-clip: padding-box;
            }}
            ::-webkit-scrollbar-thumb:hover {{ background: var(--primary-dark); }}
            
            /* Animated Medical Cross - BIG RED */
            .medical-cross {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 70px;
                height: 70px;
                background: linear-gradient(135deg, #E53935 0%, #C62828 100%);
                border-radius: 18px;
                box-shadow: 0 6px 25px rgba(229,57,53,0.5);
                animation: crossPulse 2s ease-in-out infinite;
                position: relative;
            }}
            .medical-cross::before,
            .medical-cross::after {{
                content: '';
                position: absolute;
                background: white;
                border-radius: 4px;
            }}
            .medical-cross::before {{
                width: 35px;
                height: 12px;
            }}
            .medical-cross::after {{
                width: 12px;
                height: 35px;
            }}
            @keyframes crossPulse {{
                0%, 100% {{ transform: scale(1); box-shadow: 0 6px 25px rgba(229,57,53,0.5); }}
                50% {{ transform: scale(1.1); box-shadow: 0 8px 35px rgba(229,57,53,0.7); }}
            }}
            
            /* AI Assistant Animation - Cool Orb Effect */
            .ai-avatar {{
                width: 55px;
                height: 55px;
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                animation: aiFloat 4s ease-in-out infinite;
                box-shadow: 0 4px 20px rgba(0,105,92,0.4);
            }}
            .ai-avatar::before {{
                content: '';
                width: 22px;
                height: 22px;
                background: radial-gradient(circle, white 0%, rgba(255,255,255,0.6) 100%);
                border-radius: 50%;
                animation: aiPulse 2s ease-in-out infinite;
            }}
            .ai-avatar::after {{
                content: '';
                position: absolute;
                width: 100%;
                height: 100%;
                border: 3px solid rgba(77,182,172,0.6);
                border-radius: 50%;
                animation: aiRipple 2s ease-out infinite;
            }}
            @keyframes aiFloat {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-5px); }}
            }}
            @keyframes aiPulse {{
                0%, 100% {{ opacity: 0.7; transform: scale(1); }}
                50% {{ opacity: 1; transform: scale(1.3); }}
            }}
            @keyframes aiRipple {{
                0% {{ transform: scale(1); opacity: 0.6; }}
                100% {{ transform: scale(1.4); opacity: 0; }}
            }}
            
            /* Suggestion Chips - Premium Style */
            .suggestion-chip {{
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(10px);
                border: 2px solid var(--primary-light);
                border-radius: 30px;
                padding: 12px 22px;
                cursor: pointer;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                font-size: 13px;
                font-weight: 600;
                color: var(--primary);
                box-shadow: var(--shadow-soft);
                position: relative;
                overflow: hidden;
            }}
            .suggestion-chip::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
                transition: 0.6s;
            }}
            .suggestion-chip:hover {{
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
                color: white;
                border-color: var(--primary);
                transform: translateY(-5px) scale(1.03);
                box-shadow: var(--shadow-medium);
            }}
            .suggestion-chip:hover::before {{
                left: 100%;
            }}
            
            /* Chat Bubble Animations - Smoother */
            .chat-bubble {{
                animation: bubbleIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            }}
            @keyframes bubbleIn {{
                0% {{ 
                    opacity: 0; 
                    transform: translateY(30px) scale(0.9); 
                }}
                60% {{
                    transform: translateY(-5px) scale(1.02);
                }}
                100% {{ 
                    opacity: 1; 
                    transform: translateY(0) scale(1); 
                }}
            }}
            
            /* Send Button Premium */
            .send-btn {{
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                position: relative;
                overflow: hidden;
            }}
            .send-btn::before {{
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                background: rgba(255,255,255,0.2);
                border-radius: 50%;
                transform: translate(-50%, -50%);
                transition: 0.5s;
            }}
            .send-btn:hover {{
                transform: translateY(-4px) scale(1.02);
                box-shadow: 0 10px 30px rgba(0,105,92,0.4);
            }}
            .send-btn:hover::before {{
                width: 300px;
                height: 300px;
            }}
            .send-btn:active {{
                transform: translateY(-2px) scale(0.98);
            }}
            
            /* Input Focus Effect */
            .chat-input {{
                transition: all 0.3s ease;
            }}
            .chat-input:focus {{
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 4px rgba(0,105,92,0.15), 0 4px 20px rgba(0,105,92,0.1) !important;
                outline: none;
            }}
            
            /* Location Button */
            .location-btn {{
                background: linear-gradient(135deg, #1976D2 0%, #42A5F5 100%);
                border: none;
                border-radius: 20px;
                padding: 14px 28px;
                cursor: pointer;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                color: white;
                font-weight: 600;
                font-size: 0.95rem;
                display: flex;
                align-items: center;
                gap: 10px;
                box-shadow: 0 4px 20px rgba(25,118,210,0.35);
            }}
            .location-btn:hover {{
                transform: translateY(-4px) scale(1.02);
                box-shadow: 0 8px 30px rgba(25,118,210,0.45);
            }}
            
            /* Glassmorphism Card */
            .glass-card {{
                background: rgba(255,255,255,0.88);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border-radius: 28px;
                border: 1px solid rgba(255,255,255,0.6);
                box-shadow: var(--shadow-medium);
            }}
            
            /* Status Dot Animation */
            .status-dot {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
                animation: statusGlow 2s ease-in-out infinite;
            }}
            .status-dot.active {{ 
                background: #69F0AE;
                box-shadow: 0 0 10px #69F0AE;
            }}
            .status-dot.inactive {{ 
                background: #FF5252;
                box-shadow: 0 0 10px #FF5252;
            }}
            @keyframes statusGlow {{
                0%, 100% {{ opacity: 1; transform: scale(1); }}
                50% {{ opacity: 0.7; transform: scale(0.9); }}
            }}
            
            /* Page Transitions */
            .page-fade {{
                animation: pageFade 0.5s ease-out;
            }}
            @keyframes pageFade {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            
            /* Map Modal Styles */
            .map-modal {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.7);
                backdrop-filter: blur(8px);
                z-index: 9999;
                animation: modalFadeIn 0.3s ease;
            }}
            .map-modal.active {{
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            @keyframes modalFadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            .map-container {{
                width: 90%;
                max-width: 900px;
                height: 75vh;
                background: white;
                border-radius: 24px;
                overflow: hidden;
                box-shadow: 0 25px 80px rgba(0,0,0,0.4);
                animation: modalSlideIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            }}
            @keyframes modalSlideIn {{
                from {{ transform: translateY(50px) scale(0.9); opacity: 0; }}
                to {{ transform: translateY(0) scale(1); opacity: 1; }}
            }}
            .map-header {{
                background: linear-gradient(135deg, #00695C 0%, #00897B 100%);
                color: white;
                padding: 20px 25px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .map-header h3 {{
                margin: 0;
                font-size: 1.3rem;
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            .close-map {{
                background: rgba(255,255,255,0.2);
                border: none;
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 1.5rem;
                transition: all 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .close-map:hover {{
                background: rgba(255,255,255,0.3);
                transform: rotate(90deg);
            }}
            #pharmacy-map {{
                width: 100%;
                height: calc(100% - 70px);
            }}
            .pharmacy-marker {{
                background: linear-gradient(135deg, #E53935 0%, #C62828 100%);
                border: 3px solid white;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }}
            .user-marker {{
                background: linear-gradient(135deg, #1976D2 0%, #42A5F5 100%);
                border: 3px solid white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                animation: userPulse 2s ease-in-out infinite;
            }}
            @keyframes userPulse {{
                0%, 100% {{ box-shadow: 0 0 0 0 rgba(25,118,210,0.5); }}
                50% {{ box-shadow: 0 0 0 15px rgba(25,118,210,0); }}
            }}
            .leaflet-popup-content-wrapper {{
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            }}
            .leaflet-popup-content {{
                margin: 12px 15px;
                font-family: 'Inter', sans-serif;
            }}
            .pharmacy-popup {{
                text-align: center;
            }}
            .pharmacy-popup h4 {{
                margin: 0 0 8px 0;
                color: #00695C;
                font-size: 1rem;
            }}
            .pharmacy-popup p {{
                margin: 0;
                color: #666;
                font-size: 0.85rem;
            }}
            .pharmacy-popup .directions-btn {{
                background: linear-gradient(135deg, #00695C 0%, #00897B 100%);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 20px;
                margin-top: 10px;
                cursor: pointer;
                font-size: 0.85rem;
                font-weight: 600;
                transition: all 0.3s;
            }}
            .pharmacy-popup .directions-btn:hover {{
                transform: scale(1.05);
                box-shadow: 0 4px 15px rgba(0,105,92,0.4);
            }}
            .map-loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                color: #00695C;
            }}
            .map-loading .spinner {{
                width: 50px;
                height: 50px;
                border: 4px solid #B2DFDB;
                border-top: 4px solid #00695C;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }}
            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}
            
            /* ==================== EMERGENCY MODE ==================== */
            .emergency-btn {{
                background: linear-gradient(135deg, #B71C1C 0%, #E53935 100%);
                border: 3px solid #FFCDD2;
                border-radius: 16px;
                padding: 14px 28px;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                color: white;
                font-weight: 700;
                font-size: 1rem;
                display: flex;
                align-items: center;
                gap: 10px;
                box-shadow: 0 6px 25px rgba(183,28,28,0.5);
                animation: emergencyGlow 1.5s ease-in-out infinite;
            }}
            @keyframes emergencyGlow {{
                0%, 100% {{ box-shadow: 0 6px 25px rgba(183,28,28,0.5); }}
                50% {{ box-shadow: 0 6px 40px rgba(183,28,28,0.8); }}
            }}
            .emergency-btn:hover {{
                transform: scale(1.05);
                box-shadow: 0 10px 40px rgba(183,28,28,0.7);
            }}
            
            /* Emergency Modal */
            .emergency-modal {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, rgba(183,28,28,0.95) 0%, rgba(198,40,40,0.95) 100%);
                backdrop-filter: blur(10px);
                z-index: 99999;
                animation: emergencyFadeIn 0.3s ease;
            }}
            .emergency-modal.active {{
                display: flex;
                flex-direction: column;
            }}
            @keyframes emergencyFadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            .emergency-header {{
                background: rgba(0,0,0,0.3);
                padding: 20px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 3px solid rgba(255,255,255,0.2);
            }}
            .emergency-title {{
                color: white;
                font-size: 2rem;
                font-weight: 800;
                display: flex;
                align-items: center;
                gap: 15px;
                animation: emergencyPulse 1s ease-in-out infinite;
            }}
            @keyframes emergencyPulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.02); }}
            }}
            .emergency-close {{
                background: rgba(255,255,255,0.2);
                border: 2px solid white;
                color: white;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 1.8rem;
                transition: all 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .emergency-close:hover {{
                background: white;
                color: #B71C1C;
            }}
            .emergency-content {{
                flex: 1;
                display: flex;
                gap: 20px;
                padding: 20px;
                overflow: hidden;
            }}
            .emergency-map-section {{
                flex: 1;
                background: white;
                border-radius: 20px;
                overflow: hidden;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }}
            #emergency-map {{
                width: 100%;
                height: 100%;
            }}
            .emergency-list-section {{
                width: 380px;
                display: flex;
                flex-direction: column;
                gap: 15px;
                overflow-y: auto;
                padding-right: 10px;
            }}
            .hospital-card {{
                background: rgba(255,255,255,0.95);
                border-radius: 16px;
                padding: 18px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);
                animation: hospitalSlideIn 0.5s ease;
                cursor: pointer;
                transition: all 0.3s;
            }}
            .hospital-card:hover {{
                transform: translateY(-3px);
                box-shadow: 0 8px 30px rgba(0,0,0,0.3);
            }}
            @keyframes hospitalSlideIn {{
                from {{ opacity: 0; transform: translateX(30px); }}
                to {{ opacity: 1; transform: translateX(0); }}
            }}
            .hospital-name {{
                font-size: 1.1rem;
                font-weight: 700;
                color: #B71C1C;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .hospital-info {{
                font-size: 0.85rem;
                color: #666;
                margin-bottom: 5px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .hospital-distance {{
                background: linear-gradient(135deg, #B71C1C 0%, #E53935 100%);
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
                display: inline-block;
                margin-top: 8px;
            }}
            .call-btn {{
                background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 600;
                font-size: 0.9rem;
                margin-top: 10px;
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                transition: all 0.3s;
            }}
            .call-btn:hover {{
                transform: scale(1.02);
                box-shadow: 0 4px 15px rgba(46,125,50,0.4);
            }}
            .emergency-numbers {{
                background: rgba(0,0,0,0.3);
                padding: 15px 25px;
                display: flex;
                justify-content: center;
                gap: 40px;
            }}
            .emergency-number {{
                color: white;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
            }}
            .emergency-number:hover {{
                transform: scale(1.1);
            }}
            .emergency-number .number {{
                font-size: 2rem;
                font-weight: 800;
                text-shadow: 0 2px 10px rgba(0,0,0,0.3);
            }}
            .emergency-number .label {{
                font-size: 0.8rem;
                opacity: 0.9;
            }}
            
            /* Hospital Marker */
            .hospital-marker {{
                background: linear-gradient(135deg, #B71C1C 0%, #E53935 100%);
                border: 3px solid white;
                border-radius: 50%;
                width: 35px;
                height: 35px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.4);
                animation: hospitalPing 2s ease-in-out infinite;
            }}
            @keyframes hospitalPing {{
                0%, 100% {{ box-shadow: 0 0 0 0 rgba(183,28,28,0.6); }}
                50% {{ box-shadow: 0 0 0 15px rgba(183,28,28,0); }}
            }}
            
            /* Analytics Graph Styles */
            .analytics-container {{
                background: rgba(255,255,255,0.95);
                border-radius: 16px;
                padding: 15px;
                margin-top: 15px;
                box-shadow: 0 4px 20px rgba(0,105,92,0.15);
                animation: graphSlideIn 0.5s ease;
            }}
            @keyframes graphSlideIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .analytics-title {{
                font-size: 1rem;
                font-weight: 700;
                color: #00695C;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .analytics-stats {{
                display: flex;
                gap: 15px;
                margin-top: 10px;
            }}
            .stat-card {{
                flex: 1;
                background: linear-gradient(135deg, #E0F2F1 0%, #B2DFDB 100%);
                border-radius: 12px;
                padding: 12px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 1.5rem;
                font-weight: 800;
                color: #00695C;
            }}
            .stat-label {{
                font-size: 0.75rem;
                color: #4DB6AC;
                font-weight: 500;
            }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        {{%config%}}
        {{%scripts%}}
        {{%renderer%}}
        
        <script>
            // Smooth scroll function
            function scrollToBottom() {{
                var chatDiv = document.getElementById("chat-history");
                if (chatDiv) {{
                    chatDiv.scrollTo({{
                        top: chatDiv.scrollHeight,
                        behavior: 'smooth'
                    }});
                }}
            }}
            
            // Map variables
            var pharmacyMap = null;
            var userLat = null;
            var userLng = null;
            
            // Open map modal and find pharmacies
            function openPharmacyMap() {{
                var modal = document.getElementById('map-modal');
                modal.classList.add('active');
                
                // Initialize map if not already done
                setTimeout(function() {{
                    if (!pharmacyMap) {{
                        initializeMap();
                    }}
                }}, 100);
            }}
            
            // Close map modal
            function closePharmacyMap() {{
                var modal = document.getElementById('map-modal');
                modal.classList.remove('active');
            }}
            
            // Close on background click
            function closeOnBackground(event) {{
                if (event.target.id === 'map-modal') {{
                    closePharmacyMap();
                }}
            }}
            
            // Initialize Leaflet map
            function initializeMap() {{
                // Default to a central location
                pharmacyMap = L.map('pharmacy-map').setView([20.5937, 78.9629], 5);
                
                // Add OpenStreetMap tiles (FREE!)
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                    maxZoom: 19
                }}).addTo(pharmacyMap);
                
                // Get user location
                if (navigator.geolocation) {{
                    navigator.geolocation.getCurrentPosition(function(position) {{
                        userLat = position.coords.latitude;
                        userLng = position.coords.longitude;
                        
                        // Center map on user
                        pharmacyMap.setView([userLat, userLng], 15);
                        
                        // Add user marker
                        var userIcon = L.divIcon({{
                            className: 'user-marker',
                            iconSize: [20, 20]
                        }});
                        L.marker([userLat, userLng], {{icon: userIcon}})
                            .addTo(pharmacyMap)
                            .bindPopup('<div class="pharmacy-popup"><h4>📍 You are here</h4></div>');
                        
                        // Search for nearby pharmacies using Overpass API (FREE!)
                        searchNearbyPharmacies(userLat, userLng);
                        
                    }}, function(error) {{
                        alert('Please enable location access to find nearby pharmacies.');
                    }}, {{
                        enableHighAccuracy: true,
                        timeout: 10000
                    }});
                }}
            }}
            
            // Search pharmacies using Overpass API (FREE OpenStreetMap data)
            function searchNearbyPharmacies(lat, lng) {{
                var radius = 3000; // 3km radius
                var query = `
                    [out:json][timeout:25];
                    (
                        node["amenity"="pharmacy"](around:${{radius}},${{lat}},${{lng}});
                        node["shop"="chemist"](around:${{radius}},${{lat}},${{lng}});
                        node["healthcare"="pharmacy"](around:${{radius}},${{lat}},${{lng}});
                    );
                    out body;
                `;
                
                fetch('https://overpass-api.de/api/interpreter', {{
                    method: 'POST',
                    body: query
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.elements && data.elements.length > 0) {{
                        addPharmacyMarkers(data.elements);
                    }} else {{
                        // Show message if no pharmacies found
                        L.popup()
                            .setLatLng([lat, lng])
                            .setContent('<div class="pharmacy-popup"><h4>No pharmacies found nearby</h4><p>Try zooming out or searching a different area</p></div>')
                            .openOn(pharmacyMap);
                    }}
                }})
                .catch(error => {{
                    console.log('Pharmacy search error:', error);
                }});
            }}
            
            // Add pharmacy markers to map
            function addPharmacyMarkers(pharmacies) {{
                var pharmacyIcon = L.divIcon({{
                    className: 'pharmacy-marker',
                    html: '💊',
                    iconSize: [30, 30]
                }});
                
                pharmacies.forEach(function(pharmacy) {{
                    var name = pharmacy.tags.name || 'Pharmacy';
                    var address = pharmacy.tags['addr:street'] || '';
                    
                    var popupContent = `
                        <div class="pharmacy-popup">
                            <h4>🏥 ${{name}}</h4>
                            <p>${{address}}</p>
                            <button class="directions-btn" onclick="getDirections(${{pharmacy.lat}}, ${{pharmacy.lon}})">
                                🗺️ Get Directions
                            </button>
                        </div>
                    `;
                    
                    L.marker([pharmacy.lat, pharmacy.lon], {{icon: pharmacyIcon}})
                        .addTo(pharmacyMap)
                        .bindPopup(popupContent);
                }});
            }}
            
            // Open directions in Google Maps
            function getDirections(destLat, destLng) {{
                var url = 'https://www.google.com/maps/dir/?api=1';
                if (userLat && userLng) {{
                    url += '&origin=' + userLat + ',' + userLng;
                }}
                url += '&destination=' + destLat + ',' + destLng;
                url += '&travelmode=driving';
                window.open(url, '_blank');
            }}
            
            // Create map modal on page load
            document.addEventListener('DOMContentLoaded', function() {{
                var modalHTML = `
                    <div id="map-modal" class="map-modal" onclick="if(event.target.id==='map-modal')closePharmacyMap()">
                        <div class="map-container">
                            <div class="map-header">
                                <h3>&#x1F5FA; Nearby Pharmacies - FREE</h3>
                                <button class="close-map" onclick="closePharmacyMap()">&#x2715;</button>
                            </div>
                            <div id="pharmacy-map"></div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHTML);
                
                // Check if user is already logged in
                var savedUser = localStorage.getItem('curebot_user');
                if (savedUser) {{
                    var user = JSON.parse(savedUser);
                    showMainApp(user);
                }}
            }});
            
            // ==================== USER SESSION ====================
            var currentUser = null;
            
            function showMainApp(user) {{
                currentUser = user;
                document.getElementById('login-page').style.display = 'none';
                document.getElementById('main-app').style.display = 'block';
                
                // Update user info in header
                var userInfoDiv = document.getElementById('user-info-display');
                if (userInfoDiv && user) {{
                    userInfoDiv.innerHTML = `
                        <img src="${{user.picture}}" class="user-avatar" alt="Avatar" onerror="this.src='https://ui-avatars.com/api/?name=${{encodeURIComponent(user.name)}}&background=00695C&color=fff'">
                        <span class="user-name">${{user.name.split(' ')[0]}}</span>
                        <button class="logout-btn" onclick="logoutUser()">Exit</button>
                    `;
                }}
                
                // Save to localStorage
                localStorage.setItem('curebot_user', JSON.stringify(user));
            }}
            
            function skipLogin() {{
                var guestUser = {{
                    name: 'User',
                    email: 'user@curebot.app',
                    picture: 'https://ui-avatars.com/api/?name=User&background=00695C&color=fff'
                }};
                showMainApp(guestUser);
            }}
            
            function logoutUser() {{
                localStorage.removeItem('curebot_user');
                currentUser = null;
                document.getElementById('login-page').style.display = 'flex';
                document.getElementById('main-app').style.display = 'none';
            }}
            
            // Check for saved user on page load
            window.onload = function() {{
                var savedUser = localStorage.getItem('curebot_user');
                if (savedUser) {{
                    try {{
                        showMainApp(JSON.parse(savedUser));
                    }} catch(e) {{
                        localStorage.removeItem('curebot_user');
                    }}
                }}
            }};
            
            // ==================== EMERGENCY MODE ====================
            var emergencyMap = null;
            var hospitals = [];
            
            function openEmergencyMode() {{
                var modal = document.getElementById('emergency-modal');
                modal.classList.add('active');
                
                // Play emergency sound effect (optional)
                // document.getElementById('emergency-sound').play();
                
                setTimeout(function() {{
                    if (!emergencyMap) {{
                        initializeEmergencyMap();
                    }}
                }}, 100);
            }}
            
            function closeEmergencyMode() {{
                var modal = document.getElementById('emergency-modal');
                modal.classList.remove('active');
            }}
            
            function initializeEmergencyMap() {{
                emergencyMap = L.map('emergency-map').setView([20.5937, 78.9629], 5);
                
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: 'OpenStreetMap',
                    maxZoom: 19
                }}).addTo(emergencyMap);
                
                if (navigator.geolocation) {{
                    navigator.geolocation.getCurrentPosition(function(position) {{
                        var lat = position.coords.latitude;
                        var lng = position.coords.longitude;
                        
                        emergencyMap.setView([lat, lng], 14);
                        
                        // User marker
                        var userIcon = L.divIcon({{
                            className: 'user-marker',
                            iconSize: [20, 20]
                        }});
                        L.marker([lat, lng], {{icon: userIcon}})
                            .addTo(emergencyMap)
                            .bindPopup('<b>📍 Your Location</b>');
                        
                        // Search hospitals
                        searchNearbyHospitals(lat, lng);
                        
                    }}, function(error) {{
                        alert('Please enable location for emergency services.');
                    }}, {{enableHighAccuracy: true, timeout: 15000}});
                }}
            }}
            
            function searchNearbyHospitals(lat, lng) {{
                var radius = 10000; // 10km for hospitals
                var query = `
                    [out:json][timeout:30];
                    (
                        node["amenity"="hospital"](around:${{radius}},${{lat}},${{lng}});
                        way["amenity"="hospital"](around:${{radius}},${{lat}},${{lng}});
                        node["amenity"="clinic"](around:${{radius}},${{lat}},${{lng}});
                        node["healthcare"="hospital"](around:${{radius}},${{lat}},${{lng}});
                        node["emergency"="yes"](around:${{radius}},${{lat}},${{lng}});
                    );
                    out center body;
                `;
                
                document.getElementById('hospital-list').innerHTML = '<div style="text-align:center;padding:20px;color:#fff;">🔍 Searching hospitals...</div>';
                
                fetch('https://overpass-api.de/api/interpreter', {{
                    method: 'POST',
                    body: query
                }})
                .then(response => response.json())
                .then(data => {{
                    hospitals = [];
                    if (data.elements && data.elements.length > 0) {{
                        data.elements.forEach(function(h) {{
                            var hLat = h.lat || (h.center && h.center.lat);
                            var hLng = h.lon || (h.center && h.center.lon);
                            if (hLat && hLng) {{
                                var distance = calculateDistance(lat, lng, hLat, hLng);
                                hospitals.push({{
                                    name: h.tags.name || 'Hospital',
                                    lat: hLat,
                                    lng: hLng,
                                    phone: h.tags.phone || h.tags['contact:phone'] || 'Call 102',
                                    address: h.tags['addr:street'] || h.tags['addr:city'] || '',
                                    emergency: h.tags.emergency === 'yes',
                                    distance: distance
                                }});
                            }}
                        }});
                        
                        // Sort by distance
                        hospitals.sort((a, b) => a.distance - b.distance);
                        
                        // Add markers and list
                        addHospitalMarkers(hospitals, lat, lng);
                        updateHospitalList(hospitals);
                    }} else {{
                        document.getElementById('hospital-list').innerHTML = '<div style="text-align:center;padding:20px;color:#fff;">No hospitals found. Call 102 for ambulance.</div>';
                    }}
                }})
                .catch(error => {{
                    console.error('Hospital search error:', error);
                    document.getElementById('hospital-list').innerHTML = '<div style="text-align:center;padding:20px;color:#fff;">Error searching. Call 102 for emergency.</div>';
                }});
            }}
            
            function calculateDistance(lat1, lng1, lat2, lng2) {{
                var R = 6371; // km
                var dLat = (lat2 - lat1) * Math.PI / 180;
                var dLng = (lng2 - lng1) * Math.PI / 180;
                var a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                        Math.sin(dLng/2) * Math.sin(dLng/2);
                var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
                return R * c;
            }}
            
            function addHospitalMarkers(hospitals, userLat, userLng) {{
                hospitals.forEach(function(h, index) {{
                    var hospitalIcon = L.divIcon({{
                        className: 'hospital-marker',
                        html: '🏥',
                        iconSize: [35, 35]
                    }});
                    
                    var marker = L.marker([h.lat, h.lng], {{icon: hospitalIcon}})
                        .addTo(emergencyMap)
                        .bindPopup(`
                            <div class="pharmacy-popup">
                                <h4>🏥 ${{h.name}}</h4>
                                <p>${{h.address}}</p>
                                <p><b>📞 ${{h.phone}}</b></p>
                                <button class="directions-btn" onclick="getDirections(${{h.lat}}, ${{h.lng}})">
                                    🚑 Get Directions
                                </button>
                            </div>
                        `);
                    
                    if (index === 0) {{
                        marker.openPopup();
                    }}
                }});
            }}
            
            function updateHospitalList(hospitals) {{
                var html = '';
                hospitals.slice(0, 8).forEach(function(h, index) {{
                    html += `
                        <div class="hospital-card" onclick="focusHospital(${{h.lat}}, ${{h.lng}})" style="animation-delay: ${{index * 0.1}}s">
                            <div class="hospital-name">
                                🏥 ${{h.name}}
                                ${{h.emergency ? '<span style="background:#E53935;color:white;padding:2px 8px;border-radius:10px;font-size:0.7rem;">24/7 ER</span>' : ''}}
                            </div>
                            <div class="hospital-info">📍 ${{h.address || 'Address on map'}}</div>
                            <div class="hospital-info">📞 ${{h.phone}}</div>
                            <span class="hospital-distance">${{h.distance.toFixed(1)}} km away</span>
                            <button class="call-btn" onclick="event.stopPropagation(); window.open('tel:${{h.phone.replace(/\\s/g, '')}}')">
                                📞 Call Hospital
                            </button>
                        </div>
                    `;
                }});
                document.getElementById('hospital-list').innerHTML = html;
            }}
            
            function focusHospital(lat, lng) {{
                emergencyMap.setView([lat, lng], 16);
            }}
            
            function callEmergency(number) {{
                window.open('tel:' + number);
            }}
            
            // Create Emergency Modal
            document.addEventListener('DOMContentLoaded', function() {{
                var emergencyHTML = `
                    <div id="emergency-modal" class="emergency-modal">
                        <div class="emergency-header">
                            <div class="emergency-title">
                                🚨 EMERGENCY MODE
                            </div>
                            <button class="emergency-close" onclick="closeEmergencyMode()">✕</button>
                        </div>
                        <div class="emergency-content">
                            <div class="emergency-map-section">
                                <div id="emergency-map"></div>
                            </div>
                            <div class="emergency-list-section" id="hospital-list">
                                <div style="text-align:center;padding:20px;color:#fff;">
                                    Loading nearby hospitals...
                                </div>
                            </div>
                        </div>
                        <div class="emergency-numbers">
                            <div class="emergency-number" onclick="callEmergency('102')">
                                <div class="number">102</div>
                                <div class="label">Ambulance</div>
                            </div>
                            <div class="emergency-number" onclick="callEmergency('108')">
                                <div class="number">108</div>
                                <div class="label">Emergency</div>
                            </div>
                            <div class="emergency-number" onclick="callEmergency('112')">
                                <div class="number">112</div>
                                <div class="label">All Services</div>
                            </div>
                            <div class="emergency-number" onclick="callEmergency('100')">
                                <div class="number">100</div>
                                <div class="label">Police</div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', emergencyHTML);
            }});
        </script>
    </body>
</html>
'''

# =============================================================================
# 5. APP LAYOUT - Premium Design with Login Page
# =============================================================================

app.layout = html.Div([
    
    # ==================== LOGIN PAGE ====================
    html.Div(id='login-page', className='login-page', children=[
        html.Div(className='login-container', children=[
            # Logo
            html.Div(className='login-logo'),
            
            # Title
            html.H1(APP_NAME, className='login-title'),
            html.P(APP_TAGLINE, className='login-subtitle'),
            
            # Enter CureBot Button
            html.Button([
                html.Span("🚀", style={'marginRight': '10px', 'fontSize': '20px'}),
                "Enter CureBot"
            ], id='skip-login-btn', n_clicks=0, className='enter-curebot-btn', style={
                'display': 'flex',
                'background': 'linear-gradient(135deg, #00695C 0%, #00897B 100%)',
                'border': 'none', 'borderRadius': '30px',
                'padding': '16px 40px', 'fontSize': '17px', 'fontWeight': '700',
                'cursor': 'pointer', 'alignItems': 'center', 'justifyContent': 'center',
                'color': 'white', 'width': '280px',
                'boxShadow': '0 8px 25px rgba(0,105,92,0.4)', 'transition': 'all 0.3s ease',
                'textTransform': 'uppercase', 'letterSpacing': '1px'
            }),
            
            html.P("Get AI-powered medicine recommendations", style={
                'fontSize': '12px', 'color': '#666', 'marginTop': '12px'
            }),
            
            # Features
            html.Div(className='login-features', style={'marginTop': '30px'}, children=[
                html.Div(className='login-feature', children=[
                    html.Div("🤖", className='login-feature-icon'),
                    html.Div("AI Powered", className='login-feature-text')
                ]),
                html.Div(className='login-feature', children=[
                    html.Div("💊", className='login-feature-icon'),
                    html.Div("248K+ Medicines", className='login-feature-text')
                ]),
                html.Div(className='login-feature', children=[
                    html.Div("🗺️", className='login-feature-icon'),
                    html.Div("Find Hospitals", className='login-feature-text')
                ]),
            ])
        ])
    ]),
    
    # ==================== MAIN APP (Hidden initially) ====================
    html.Div(id='main-app', style={'display': 'none'}, children=[
    
        # --- Premium Header with Big Animated Medical Cross ---
        html.Div([
            html.Div([
                # Big Animated Medical Cross (Red Plus)
                html.Div(className='medical-cross', style={'marginRight': '25px'}),
                
                # Title
                html.Div([
                    html.H1(APP_NAME, style={
                        'color': 'white', 'margin': '0',
                        'fontSize': '3.2rem', 'fontWeight': '800', 
                        'letterSpacing': '3px',
                        'textShadow': '0 3px 15px rgba(0,0,0,0.2)'
                    }),
                    html.P(APP_TAGLINE, style={
                        'color': '#B2DFDB', 'fontSize': '1.15rem', 
                        'marginTop': '8px', 'fontWeight': '400',
                        'letterSpacing': '1.5px'
                    }),
                ])
            ], style={
                'display': 'flex', 'alignItems': 'center', 
            'justifyContent': 'center', 'marginBottom': '20px'
        }),
        
        # Status Badge & Find Pharmacy Button
        html.Div([
            # Status
            html.Div([
                html.Span(className='status-dot active' if DATA_LOADED else 'status-dot inactive'),
                html.Span(
                    f"  AI Active • {len(df1):,} medicines" if DATA_LOADED else "  Offline",
                    style={'color': '#B2DFDB', 'fontSize': '0.95rem', 'marginLeft': '10px', 'fontWeight': '500'}
                )
            ], style={'display': 'flex', 'alignItems': 'center'}),
            
            # Find Pharmacy Button
            html.Button([
                html.Span("📍", style={'fontSize': '1.3rem'}),
                html.Span(" Find Pharmacy")
            ], className='location-btn', id='find-pharmacy-btn', n_clicks=0),
            
            # Emergency Button
            html.Button([
                html.Span("🚨", style={'fontSize': '1.3rem'}),
                html.Span(" EMERGENCY")
            ], className='emergency-btn', id='emergency-btn', n_clicks=0),
            
            # User Info (populated by JavaScript)
            html.Div(id='user-info-display', className='user-info', style={'display': 'flex'})
            
        ], style={
            'display': 'flex', 'alignItems': 'center', 
            'justifyContent': 'center', 'gap': '35px', 'flexWrap': 'wrap'
        })
        
    ], style={
        'background': 'linear-gradient(135deg, #004D40 0%, #00695C 30%, #00897B 70%, #4DB6AC 100%)',
        'padding': '45px 25px 40px',
        'borderRadius': '0 0 45px 45px',
        'marginBottom': '35px',
        'textAlign': 'center',
        'boxShadow': '0 10px 50px rgba(0,77,64,0.35)'
    }),

    # --- Main Chat Container ---
    html.Div([
        # Chat Box with Glass Effect
        html.Div(id='chat-history', className='glass-card', children=[
            # Welcome Message with AI Avatar
            html.Div([
                html.Div(className='ai-avatar', style={'marginRight': '18px', 'flexShrink': '0'}),
                html.Div([
                    html.Div(f"Welcome to {APP_NAME}!", style={
                        'fontWeight': '700', 'fontSize': '1.15rem', 
                        'color': 'var(--primary-dark)', 'marginBottom': '8px'
                    }),
                    html.Div("I'm your AI medicine assistant. Describe your symptoms and I'll help you find the right medicine. Use the quick buttons below or type in detail!", 
                             style={'lineHeight': '1.7', 'color': 'var(--primary)', 'fontSize': '0.95rem'})
                ])
            ], className='chat-bubble', style={
                'display': 'flex', 'alignItems': 'flex-start',
                'background': 'linear-gradient(135deg, rgba(224,242,241,0.95) 0%, rgba(178,223,219,0.95) 100%)',
                'padding': '22px 25px',
                'borderRadius': '12px 28px 28px 28px',
                'marginBottom': '20px',
                'border': '1px solid rgba(77,182,172,0.4)'
            })
        ], style={
            'height': '460px',
            'overflowY': 'auto',
            'padding': '28px',
            'marginBottom': '25px',
        }),

        # Quick Symptom Buttons - More Options & Better Layout
        html.Div([
            html.Div([
                html.Span("⚡", style={'fontSize': '1.3rem'}),
                html.Span(" Quick Search:", style={'fontWeight': '700', 'color': 'var(--primary-dark)', 'marginLeft': '8px'})
            ], style={'marginRight': '15px', 'display': 'flex', 'alignItems': 'center'}),
        ] + [
            html.Button(f"{emoji} {label}", id=f'btn-{id_name}', n_clicks=0, className='suggestion-chip')
            for id_name, label, emoji in [
                ('headache', 'Headache', '🤕'),
                ('fever', 'Fever', '🌡️'),
                ('cold', 'Cold & Flu', '🤧'),
                ('cough', 'Cough', '😷'),
                ('pain', 'Body Pain', '💪'),
                ('nausea', 'Nausea', '🤢'),
                ('sleep', 'Sleep', '😴'),
                ('allergy', 'Allergy', '🤧'),
                ('diabetes', 'Diabetes', '🩸'),
                ('bp', 'Blood Pressure', '❤️'),
                ('acidity', 'Acidity', '🔥'),
                ('skin', 'Skin Issues', '🧴'),
                ('vitamin', 'Vitamins', '💊'),
                ('anxiety', 'Anxiety', '😰'),
            ]
        ], style={
            'marginBottom': '28px', 'display': 'flex', 'gap': '10px',
            'flexWrap': 'wrap', 'alignItems': 'center', 'justifyContent': 'center'
        }),

        # Input Area - Premium Design
        html.Div([
            dcc.Input(
                id='user-input',
                type='text',
                placeholder='💬 Describe your symptoms in detail...',
                className='chat-input',
                style={
                    'flex': '1', 'padding': '20px 28px', 'borderRadius': '35px',
                    'border': '2px solid var(--primary-light)', 'fontSize': '1rem',
                    'background': 'rgba(255,255,255,0.98)',
                    'boxShadow': 'var(--shadow-soft)',
                }
            ),
            html.Button([
                html.Span("🔍", style={'marginRight': '10px', 'fontSize': '1.2rem'}),
                html.Span("Find Medicine")
            ],
                id='send-btn',
                n_clicks=0,
                className='send-btn',
                style={
                    'padding': '20px 40px',
                    'background': 'linear-gradient(135deg, #00695C 0%, #00897B 100%)',
                    'color': 'white', 'border': 'none', 'borderRadius': '35px',
                    'cursor': 'pointer', 'fontWeight': '700', 'fontSize': '1.05rem',
                    'boxShadow': '0 6px 25px rgba(0,105,92,0.4)',
                    'display': 'flex', 'alignItems': 'center'
                }
            )
        ], style={'display': 'flex', 'gap': '18px'})

    ], style={
        'maxWidth': '1050px', 'margin': '0 auto', 'padding': '0 28px',
        'paddingBottom': '140px'
    }),

    # --- Premium Footer ---
    html.Footer([
        html.Div([
            html.Span("⚕️", style={'fontSize': '1.4rem', 'marginRight': '15px'}),
            html.Span([
                html.Strong("Disclaimer: "),
                f"{APP_NAME} is for educational purposes only. ",
                html.Strong("Always consult a qualified doctor. "),
                "Emergency? Call 102 / 108 / 911"
            ], style={'opacity': '0.95'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
    ], style={
        'position': 'fixed', 'bottom': '0', 'width': '100%',
        'background': 'linear-gradient(135deg, #004D40 0%, #00695C 100%)',
        'color': '#E0F2F1', 'textAlign': 'center', 'padding': '20px 28px',
        'fontSize': '0.95rem', 'boxShadow': '0 -6px 30px rgba(0,0,0,0.2)',
        'zIndex': '1000'
    }),

    ]),  # End of main-app div

    # --- Hidden Components ---
    dcc.Store(id='store-conversation', data=[]),
    dcc.Store(id='store-user', data=None),
    html.Div(id='dummy-scroll-trigger', style={'display': 'none'}),
    html.Div(id='pharmacy-trigger', style={'display': 'none'}),
    html.Div(id='skip-login-trigger', style={'display': 'none'}),
    html.Div(id='fallback-google-trigger', style={'display': 'none'}),
    html.Div(id='emergency-trigger', style={'display': 'none'})

])

# =============================================================================
# 6. CALLBACKS
# =============================================================================

# Auto-scroll with smooth animation
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var chatDiv = document.getElementById("chat-history");
            if (chatDiv) {
                chatDiv.scrollTo({top: chatDiv.scrollHeight, behavior: 'smooth'});
            }
        }, 150);
        return "";
    }
    """,
    Output('dummy-scroll-trigger', 'children'),
    Input('chat-history', 'children')
)

# Find Pharmacy - Opens the embedded map modal
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            openPharmacyMap();
        }
        return "";
    }
    """,
    Output('pharmacy-trigger', 'children'),
    Input('find-pharmacy-btn', 'n_clicks')
)

# Skip Login - Skips Google Sign-In
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            skipLogin();
        }
        return "";
    }
    """,
    Output('skip-login-trigger', 'children'),
    Input('skip-login-btn', 'n_clicks')
)

# Fallback Google Button - Acts as Skip Login when Google fails
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            skipLogin();
        }
        return "";
    }
    """,
    Output('fallback-google-trigger', 'children'),
    Input('fallback-google-btn', 'n_clicks')
)

# Emergency Mode - Opens Emergency Modal
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            openEmergencyMode();
        }
        return "";
    }
    """,
    Output('emergency-trigger', 'children'),
    Input('emergency-btn', 'n_clicks')
)

# Main Chat Callback
@app.callback(
    [Output('chat-history', 'children'),
     Output('store-conversation', 'data'),
     Output('user-input', 'value')],
    [Input('send-btn', 'n_clicks'),
     Input('user-input', 'n_submit')] + 
    [Input(f'btn-{id_name}', 'n_clicks') for id_name in [
        'headache', 'fever', 'cold', 'cough', 'pain', 'nausea', 
        'sleep', 'allergy', 'diabetes', 'bp', 'acidity', 'skin', 'vitamin', 'anxiety'
    ]],
    [State('user-input', 'value'),
     State('store-conversation', 'data')],
    prevent_initial_call=True
)
def update_chat(n_clicks, n_submit, *args):
    # Get button clicks and states
    btn_clicks = args[:-2]
    user_text, conversation = args[-2], args[-1]
    
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Symptom mapping with expanded terms
    symptom_map = {
        'btn-headache': ('Headache', 'headache migraine head pain tension'),
        'btn-fever': ('Fever', 'fever high temperature pyrexia'),
        'btn-cold': ('Cold & Flu', 'cold flu influenza runny nose sneezing'),
        'btn-cough': ('Cough', 'cough bronchitis dry cough wet cough'),
        'btn-pain': ('Body Pain', 'pain body ache muscle pain joint pain arthritis'),
        'btn-nausea': ('Nausea', 'nausea vomiting stomach upset'),
        'btn-sleep': ('Sleep Issues', 'insomnia sleep disorder sleeplessness'),
        'btn-allergy': ('Allergy', 'allergy allergic reaction itching hives antihistamine'),
        'btn-diabetes': ('Diabetes', 'diabetes blood sugar glucose antidiabetic'),
        'btn-bp': ('Blood Pressure', 'hypertension high blood pressure bp antihypertensive'),
        'btn-acidity': ('Acidity', 'acidity heartburn acid reflux gastritis antacid'),
        'btn-skin': ('Skin Issues', 'skin rash eczema dermatitis cream ointment'),
        'btn-vitamin': ('Vitamins', 'vitamin supplement deficiency multivitamin'),
        'btn-anxiety': ('Anxiety', 'anxiety stress tension nervousness anxiolytic'),
    }
    
    final_text = ""
    display_text = ""
    
    if trigger_id == 'send-btn' or trigger_id == 'user-input':
        final_text = user_text
        display_text = user_text
    elif trigger_id in symptom_map:
        display_text, final_text = symptom_map[trigger_id]
    
    if not final_text:
        return dash.no_update, dash.no_update, dash.no_update

    # Emergency check
    emergency_keywords = ["heart attack", "stroke", "chest pain", "breathing difficulty", 
                         "unconscious", "severe bleeding", "poisoning", "overdose", "suicide"]
    is_emergency = any(k in final_text.lower() for k in emergency_keywords)

    conversation.append({'role': 'user', 'content': display_text, 'time': datetime.now().strftime("%H:%M")})
    
    page_style = {
        'minHeight': '100vh',
        'background': 'var(--bg-gradient)',
        'transition': 'all 0.6s ease'
    }

    # Generate Response
    if is_emergency:
        response_text = "🚨 EMERGENCY! Call ambulance immediately: 102 / 108 / 911. Do NOT wait for online advice!"
        page_style['background'] = 'linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%)'
        conversation.append({'role': 'ai', 'content': response_text, 'data': None, 'is_emergency': True, 'gemini_advice': None})
    elif not DATA_LOADED:
        response_text = "❌ System Error: Database unavailable. Please try again later."
        conversation.append({'role': 'ai', 'content': response_text, 'data': None, 'is_emergency': False, 'gemini_advice': None})
    else:
        recs, error_msg = get_ai_recommendation(final_text)
        
        # Get Gemini AI advice (if API key configured and we have results)
        gemini_advice = None
        if recs and GEMINI_ENABLED:
            gemini_advice = get_gemini_health_advice(final_text, recs)
        
        if error_msg:
            response_text = f"🤖 {error_msg}"
            conversation.append({'role': 'ai', 'content': response_text, 'data': None, 'is_emergency': False, 'gemini_advice': None})
        elif recs:
            response_text = f"✅ Found {len(recs)} medicines matching your symptoms:"
            conversation.append({'role': 'ai', 'content': response_text, 'data': recs, 'is_emergency': False, 'gemini_advice': gemini_advice})
        else:
            response_text = "😔 No exact matches found. Try different keywords or describe symptoms in more detail."
            conversation.append({'role': 'ai', 'content': response_text, 'data': None, 'is_emergency': False, 'gemini_advice': None})

    # Render Chat Bubbles
    chat_bubbles = []
    
    # Welcome message with AI avatar
    chat_bubbles.append(html.Div([
        html.Div(className='ai-avatar', style={'marginRight': '18px', 'flexShrink': '0'}),
        html.Div([
            html.Div(f"Welcome to {APP_NAME}!", style={
                'fontWeight': '700', 'fontSize': '1.15rem',
                'color': 'var(--primary-dark)', 'marginBottom': '8px'
            }),
            html.Div("I'm your AI medicine assistant. Tell me your symptoms!", 
                     style={'lineHeight': '1.7', 'color': 'var(--primary)', 'fontSize': '0.95rem'})
        ])
    ], className='chat-bubble', style={
        'display': 'flex', 'alignItems': 'flex-start',
        'background': 'linear-gradient(135deg, rgba(224,242,241,0.95) 0%, rgba(178,223,219,0.95) 100%)',
        'padding': '22px 25px',
        'borderRadius': '12px 28px 28px 28px',
        'marginBottom': '20px',
        'border': '1px solid rgba(77,182,172,0.4)'
    }))
    
    for msg in conversation:
        if msg['role'] == 'user':
            # User bubble
            chat_bubbles.append(html.Div([
                html.Span(msg['content']),
                html.Span(msg.get('time', ''), style={
                    'fontSize': '0.75rem', 'opacity': '0.7', 
                    'marginLeft': '12px'
                })
            ], className='chat-bubble', style={
                'background': 'linear-gradient(135deg, #00695C 0%, #00897B 100%)',
                'color': 'white', 'padding': '16px 24px',
                'borderRadius': '28px 28px 8px 28px',
                'marginBottom': '16px', 'maxWidth': '75%', 'marginLeft': 'auto',
                'boxShadow': '0 5px 20px rgba(0,105,92,0.3)',
                'fontWeight': '500', 'display': 'flex', 'alignItems': 'center', 'gap': '8px'
            }))
        else:
            is_msg_emergency = msg.get('is_emergency', False)
            
            if is_msg_emergency:
                bubble_bg = 'linear-gradient(135deg, #E53935 0%, #C62828 100%)'
                bubble_border = '2px solid #B71C1C'
                text_color = 'white'
            else:
                bubble_bg = 'linear-gradient(135deg, rgba(224,242,241,0.98) 0%, rgba(178,223,219,0.98) 100%)'
                bubble_border = '1px solid rgba(77,182,172,0.4)'
                text_color = 'var(--primary-dark)'
            
            chat_bubbles.append(html.Div([
                html.Div(className='ai-avatar', style={
                    'marginRight': '18px', 'flexShrink': '0',
                    'width': '45px', 'height': '45px'
                }) if not is_msg_emergency else html.Span("🚨", style={'fontSize': '2.2rem', 'marginRight': '18px'}),
                html.Span(msg['content'], style={'color': text_color, 'fontWeight': '500'})
            ], className='chat-bubble', style={
                'display': 'flex', 'alignItems': 'center',
                'background': bubble_bg,
                'padding': '20px 25px',
                'borderRadius': '12px 28px 28px 28px',
                'marginBottom': '16px', 'maxWidth': '82%',
                'border': bubble_border,
                'boxShadow': '0 5px 20px rgba(0,0,0,0.1)'
            }))
            
            # Medicine table with premium styling
            if msg.get('data'):
                df = pd.DataFrame(msg['data'])
                display_cols = ['Medicine Name', 'Primary Use', 'Form', 'Class', 'Type']
                df_display = df[[c for c in display_cols if c in df.columns]]
                
                chat_bubbles.append(html.Div(
                    dash_table.DataTable(
                        data=df_display.to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in df_display.columns],
                        style_cell={
                            'textAlign': 'left', 'fontFamily': 'Inter, sans-serif',
                            'padding': '15px 20px', 'fontSize': '0.92rem',
                            'border': 'none', 'borderBottom': '1px solid rgba(0,105,92,0.1)'
                        },
                        style_header={
                            'fontWeight': '700',
                            'background': 'linear-gradient(135deg, #00695C 0%, #00897B 100%)',
                            'color': 'white', 'border': 'none',
                            'padding': '18px 20px', 'fontSize': '0.95rem'
                        },
                        style_data_conditional=[
                            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(224,242,241,0.6)'},
                            {'if': {'row_index': 'even'}, 'backgroundColor': 'white'}
                        ],
                        style_table={'borderRadius': '22px', 'overflow': 'hidden'},
                        style_as_list_view=True
                    ), className='chat-bubble glass-card', style={
                        'maxWidth': '100%', 'marginBottom': '22px',
                        'boxShadow': '0 8px 30px rgba(0,105,92,0.18)', 
                        'borderRadius': '22px', 'overflow': 'hidden'
                    }
                ))
                
                # Add AI Disease Analytics Graph
                try:
                    # Get the search term for analytics
                    user_symptom = conversation[-2]['content'] if len(conversation) >= 2 else 'cold'
                    fig, stats = create_disease_analytics_graph(user_symptom)
                    
                    chat_bubbles.append(html.Div([
                        html.Div([
                            html.Span("📊", style={'fontSize': '1.2rem'}),
                            html.Span(" AI Disease Analytics", style={'fontWeight': '700'})
                        ], className='analytics-title'),
                        
                        dcc.Graph(
                            figure=fig,
                            config={'displayModeBar': False, 'responsive': True},
                            style={'height': '200px'}
                        ),
                        
                        html.Div([
                            html.Div([
                                html.Div(f"{stats['avg_duration']} days", className='stat-value'),
                                html.Div("Avg Recovery", className='stat-label')
                            ], className='stat-card'),
                            html.Div([
                                html.Div(stats['severity'], className='stat-value', style={
                                    'color': '#E53935' if stats['severity'] == 'High' else '#FF9800' if stats['severity'] == 'Medium' else '#4CAF50'
                                }),
                                html.Div("Severity Level", className='stat-label')
                            ], className='stat-card'),
                            html.Div([
                                html.Div("🤖 AI", className='stat-value'),
                                html.Div("Powered by", className='stat-label')
                            ], className='stat-card'),
                        ], className='analytics-stats')
                    ], className='analytics-container'))
                except Exception as e:
                    print(f"Analytics error: {e}")
                
                # Add Gemini AI Health Advice (if available)
                gemini_advice = msg.get('gemini_advice')
                if gemini_advice:
                    chat_bubbles.append(html.Div([
                        html.Div([
                            html.Img(src='https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg', 
                                    style={'width': '24px', 'height': '24px', 'marginRight': '10px'}),
                            html.Span("Google Gemini AI Advice", style={
                                'fontWeight': '700', 'color': '#1a73e8', 'fontSize': '1rem'
                            })
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
                        html.P(gemini_advice, style={
                            'color': '#333', 'lineHeight': '1.7', 'fontSize': '0.95rem',
                            'margin': '0', 'padding': '0'
                        })
                    ], style={
                        'background': 'linear-gradient(135deg, #E8F0FE 0%, #D2E3FC 100%)',
                        'borderRadius': '16px', 'padding': '20px',
                        'marginBottom': '20px', 'border': '1px solid #4285F4',
                        'boxShadow': '0 4px 15px rgba(66,133,244,0.15)'
                    }))

    return chat_bubbles, conversation, ""

# =============================================================================
# 7. RUN THE APP
# =============================================================================

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 7860))  # HuggingFace uses 7860
    
    print("\n" + "═"*65)
    print(f"   ➕ {APP_NAME} v{APP_VERSION} - {APP_TAGLINE}")
    print("═"*65)
    print(f"   📊 ML Model: Advanced TF-IDF (n-grams 1-4)")
    print(f"   💊 Medicines: {len(df1):,}" if df1 is not None else "   💊 Medicines: 0")
    print(f"   🧠 Symptom Categories: {len(SYMPTOM_SYNONYMS)}")
    print(f"   🔋 Status: {'Active ✅' if DATA_LOADED else 'Inactive ❌'}")
    print(f"   📍 Google Maps: {'Enabled ✅' if GOOGLE_MAPS_ENABLED else 'Click button to use'}")
    print("─"*65)
    print(f"   🌐 Open: http://127.0.0.1:{port}")
    print("═"*65 + "\n")
    
    app.run(debug=False, port=port, host='0.0.0.0')
