import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import zipfile
import os

# --- Data Loading & Processing Functions ---
def extract_zip_if_needed(zip_path, extract_to='.'):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    return False

def load_data():
    # 1. Check/Load Dataset 1
    file1 = 'all_medicine databased.csv'
    if not os.path.exists(file1):
        print(f"CRITICAL ERROR: '{file1}' not found in {os.getcwd()}")
        return None, None
    
    # FIX: Added low_memory=False to prevent warnings
    df1 = pd.read_csv(file1, low_memory=False)

    # 2. Check/Load Dataset 2
    file2 = 'medicine_dataset.csv'
    zip2 = 'medicine_dataset.csv.zip'
    
    if not os.path.exists(file2):
        if os.path.exists(zip2):
            print("Unzipping dataset...")
            extract_zip_if_needed(zip2)
    
    if not os.path.exists(file2):
        print(f"CRITICAL ERROR: '{file2}' not found.")
        return None, None
        
    df2 = pd.read_csv(file2, low_memory=False)

    # Preprocessing
    # Ensure columns exist before processing
    use_cols = [c for c in df1.columns if 'use' in c]
    df1['combined_use'] = df1[use_cols].fillna('').agg(' '.join, axis=1).str.lower()
    
    if 'name' in df1.columns:
        df1['name_clean'] = df1['name'].str.lower().str.strip()
    
    if 'Name' in df2.columns:
        df2['Name_clean'] = df2['Name'].str.lower().str.strip()
    
    return df1, df2

def train_model(df1):
    # Safe guard against empty data
    if df1 is None or df1.empty:
        return None, None
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df1['combined_use'].astype(str))
    return vectorizer, tfidf_matrix

# --- Initialize Data Globally (Safe Mode) ---
print("Initializing App...")
df1, df2 = load_data()

# FIX: Only train if data loaded successfully
if df1 is not None and df2 is not None:
    vectorizer, tfidf_matrix = train_model(df1)
    DATA_LOADED = True
    print("Data Loaded & Model Trained Successfully.")
else:
    vectorizer, tfidf_matrix = None, None
    DATA_LOADED = False
    print("WARNING: App starting in 'No Data' mode.")

# --- Core Logic Functions ---
def get_recommendations(user_input):
    if not DATA_LOADED:
        return pd.DataFrame() # Return empty if no data
        
    user_vec = vectorizer.transform([user_input.lower()])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    indices = cosine_sim.argsort()[-50:][::-1]
    relevant_indices = [i for i in indices if cosine_sim[i] > 0.1]
    return df1.iloc[relevant_indices]

def intersect_datasets(candidates_df1, df2):
    if candidates_df1.empty or df2 is None:
        return []

    valid_medicines = []
    df2_names = set(df2['Name_clean'].unique())
    
    for idx, row in candidates_df1.iterrows():
        # Safety check for column existence
        if 'name' not in row: continue
            
        first_word = str(row['name']).split()[0].lower()
        matches = difflib.get_close_matches(first_word, df2_names, n=1, cutoff=0.8)
        
        if matches:
            # Safe access to df2 rows
            match_row = df2[df2['Name_clean'] == matches[0]].iloc[0]
            valid_medicines.append({
                'df1_name': row['name'],
                'df2_match': matches[0],
                'dosage_form': match_row.get('Dosage Form', 'Unknown'),
                'classification': match_row.get('Classification', 'Unknown'),
                'strength': match_row.get('Strength', 'Unknown'),
                'indication': match_row.get('Indication', 'Unknown')
            })
    return valid_medicines

# --- Dash App Setup ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Styles
chat_style = {
    'height': '400px', 'overflowY': 'scroll', 'border': '1px solid #ccc',
    'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px',
    'marginBottom': '10px', 'display': 'flex', 'flexDirection': 'column'
}
user_msg_style = {
    'backgroundColor': '#007bff', 'color': 'white', 'padding': '10px',
    'borderRadius': '15px 15px 0 15px', 'alignSelf': 'flex-end',
    'maxWidth': '70%', 'marginBottom': '10px'
}
bot_msg_style = {
    'backgroundColor': '#e9ecef', 'color': 'black', 'padding': '10px',
    'borderRadius': '15px 15px 15px 0', 'alignSelf': 'flex-start',
    'maxWidth': '70%', 'marginBottom': '10px'
}

# Layout
app.layout = html.Div([
    html.H1("🏥 MediMatch AI", style={'textAlign': 'center', 'color': '#333'}),
    
    # Store components
    dcc.Store(id='store-step', data=0),
    dcc.Store(id='store-candidates', data=[]),
    dcc.Store(id='store-history', data=[{'role': 'bot', 'content': 'Hello! Please describe your symptoms or diagnosis.'}]),

    html.Div([
        # Status Alert (Only shows if data missing)
        html.Div(
            "⚠️ WARNING: Database files missing. Please add .csv files to folder and restart.",
            style={'display': 'none' if DATA_LOADED else 'block', 'color': 'red', 'fontWeight': 'bold', 'textAlign': 'center'}
        ),

        # Chat Display
        html.Div(id='chat-display', style=chat_style),
        
        # Input Area
        html.Div([
            dcc.Input(id='user-input', type='text', placeholder='Type here...', style={'width': '80%', 'padding': '10px'}),
            html.Button('Send', id='send-btn', n_clicks=0, style={'width': '18%', 'padding': '10px', 'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'cursor': 'pointer'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
        
        # Results Table
        html.Br(),
        html.Div(id='results-area')
    ], style={'maxWidth': '800px', 'margin': '0 auto', 'fontFamily': 'Arial, sans-serif'})
])

# --- Helper Functions for Callback ---
def render_chat(history):
    messages = []
    for msg in history:
        style = user_msg_style if msg['role'] == 'user' else bot_msg_style
        messages.append(html.Div(msg['content'], style=style))
    return messages

def finalize_results(history, final_candidates):
    history.append({'role': 'bot', 'content': "Here are the recommended medicines:"})
    
    df_res = pd.DataFrame(final_candidates)
    if not df_res.empty:
        df_display = df_res[['df1_name', 'df2_match', 'dosage_form', 'classification', 'strength']]
        df_display.columns = ['Medicine Name', 'Inventory Match', 'Form', 'Class', 'Strength']
        
        table = dash_table.DataTable(
            data=df_display.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df_display.columns],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
        history.append({'role': 'bot', 'content': "Check the table below. Type anything to start over."})
        return render_chat(history), history, 4, [], "", table
    else:
        history.append({'role': 'bot', 'content': "No matches found after filtering."})
        return render_chat(history), history, 0, [], "", ""

# --- Main Callback ---
@app.callback(
    [Output('chat-display', 'children'),
     Output('store-history', 'data'),
     Output('store-step', 'data'),
     Output('store-candidates', 'data'),
     Output('user-input', 'value'),
     Output('results-area', 'children')],
    [Input('send-btn', 'n_clicks'),
     Input('user-input', 'n_submit')],
    [State('user-input', 'value'),
     State('store-history', 'data'),
     State('store-step', 'data'),
     State('store-candidates', 'data')]
)
def update_chat(n_clicks, n_submit, user_text, history, step, candidates):
    ctx = callback_context
    if not ctx.triggered:
        return render_chat(history), history, step, candidates, "", ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if (trigger_id == 'send-btn' or trigger_id == 'user-input') and user_text:
        
        # Check if data is actually loaded
        if not DATA_LOADED:
            history.append({'role': 'user', 'content': user_text})
            history.append({'role': 'bot', 'content': "❌ System Error: Database files are missing. Cannot process request."})
            return render_chat(history), history, step, candidates, "", ""

        history.append({'role': 'user', 'content': user_text})
        
        # State 0: Initial Symptom Input
        if step == 0:
            initial_res = get_recommendations(user_text)
            if initial_res.empty:
                history.append({'role': 'bot', 'content': "I couldn't find matches. Please try simpler symptoms."})
                return render_chat(history), history, 0, [], "", ""
            
            valid_cands = intersect_datasets(initial_res, df2)
            
            if not valid_cands:
                history.append({'role': 'bot', 'content': "Medicines found but not in current inventory. Try different symptoms."})
                return render_chat(history), history, 0, [], "", ""
            
            forms = list(set([c['dosage_form'] for c in valid_cands]))
            if len(forms) > 1:
                history.append({'role': 'bot', 'content': f"Found matches. Preferred Dosage Form? (Options: {', '.join(forms)})"})
                return render_chat(history), history, 2, valid_cands, "", ""
            else:
                # If only 1 form, skip to next question or finish
                classifications = list(set([c['classification'] for c in valid_cands]))
                if len(classifications) > 1:
                    history.append({'role': 'bot', 'content': f"Dosage is {forms[0]}. Preferred Class? (Options: {', '.join(classifications)})"})
                    return render_chat(history), history, 3, valid_cands, "", ""
                else:
                    return finalize_results(history, valid_cands)

        # State 2: Filtering by Dosage
        elif step == 2:
            filtered = [c for c in candidates if user_text.lower() in str(c['dosage_form']).lower()]
            if not filtered:
                history.append({'role': 'bot', 'content': "Option not found. Showing all found items."})
                filtered = candidates
            
            classifications = list(set([c['classification'] for c in filtered]))
            if len(classifications) > 1:
                history.append({'role': 'bot', 'content': f"Preferred Classification? (Options: {', '.join(classifications)})"})
                return render_chat(history), history, 3, filtered, "", ""
            else:
                return finalize_results(history, filtered)

        # State 3: Filtering by Classification
        elif step == 3:
            filtered = [c for c in candidates if user_text.lower() in str(c['classification']).lower()]
            if not filtered:
                 filtered = candidates
            return finalize_results(history, filtered)
        
        # State 4: Reset
        elif step == 4:
            history = [{'role': 'bot', 'content': 'Hello! Please describe your symptoms or diagnosis.'}]
            return render_chat(history), history, 0, [], "", ""

    return render_chat(history), history, step, candidates, "", ""

if __name__ == '__main__':
    # Use 8051 to avoid conflicts
    app.run(debug=True, port=8051)
