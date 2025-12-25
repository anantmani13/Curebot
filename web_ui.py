"""
=============================================================================
CUREBOT WEB UI - Frontend Development
=============================================================================
This file contains ALL the Web Development logic for CureBot

Team: Web Development Team (2 members)
Purpose: Premium UI/UX with Dash Framework

Technologies Used:
- Dash (Python Web Framework)
- Plotly (Interactive Graphs)
- HTML/CSS (Inline Styling)
- JavaScript (Google Maps, OAuth)
- Leaflet.js (OpenStreetMap)

=============================================================================
"""

from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go

# =============================================================================
# 1. APP CONFIGURATION
# =============================================================================

# Application Settings
APP_CONFIG = {
    'title': 'CureBot - AI Medicine Recommendation',
    'version': '3.0',
    'port': 7860,  # Hugging Face default port
    'debug': False,
    'theme_primary': '#00695C',      # Teal
    'theme_secondary': '#004D40',    # Dark Teal
    'theme_accent': '#4DB6AC',       # Light Teal
    'theme_background': 'linear-gradient(135deg, #E8F5E9 0%, #B2DFDB 50%, #E0F2F1 100%)',
    'google_client_id': '1079027064414-82gdpim62um96jjgg91tcct8oucapph.apps.googleusercontent.com'
}


# =============================================================================
# 2. CSS STYLES
# =============================================================================

STYLES = {
    # Main Container
    'container': {
        'minHeight': '100vh',
        'background': APP_CONFIG['theme_background'],
        'fontFamily': 'Inter, system-ui, -apple-system, sans-serif',
    },
    
    # Header
    'header': {
        'background': f'linear-gradient(135deg, {APP_CONFIG["theme_primary"]} 0%, {APP_CONFIG["theme_secondary"]} 100%)',
        'padding': '20px 40px',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.15)',
    },
    
    # Logo
    'logo': {
        'fontSize': '32px',
        'fontWeight': '800',
        'color': 'white',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '12px',
    },
    
    # Search Box
    'search_box': {
        'width': '100%',
        'padding': '18px 25px',
        'fontSize': '18px',
        'border': '2px solid #B2DFDB',
        'borderRadius': '16px',
        'background': 'white',
        'outline': 'none',
        'boxShadow': '0 4px 15px rgba(0,105,92,0.1)',
        'transition': 'all 0.3s ease',
    },
    
    # Primary Button
    'btn_primary': {
        'background': f'linear-gradient(135deg, {APP_CONFIG["theme_primary"]} 0%, {APP_CONFIG["theme_secondary"]} 100%)',
        'color': 'white',
        'padding': '16px 40px',
        'fontSize': '18px',
        'fontWeight': '600',
        'border': 'none',
        'borderRadius': '12px',
        'cursor': 'pointer',
        'boxShadow': '0 4px 15px rgba(0,105,92,0.3)',
        'transition': 'all 0.3s ease',
    },
    
    # Emergency Button
    'btn_emergency': {
        'background': 'linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%)',
        'color': 'white',
        'padding': '12px 24px',
        'fontSize': '16px',
        'fontWeight': '600',
        'border': 'none',
        'borderRadius': '25px',
        'cursor': 'pointer',
        'boxShadow': '0 4px 15px rgba(211,47,47,0.4)',
        'animation': 'pulse 2s infinite',
    },
    
    # Card
    'card': {
        'background': 'rgba(255,255,255,0.95)',
        'borderRadius': '16px',
        'padding': '25px',
        'boxShadow': '0 8px 32px rgba(0,0,0,0.08)',
        'marginBottom': '20px',
        'border': '1px solid rgba(0,105,92,0.1)',
    },
    
    # Medicine Card
    'medicine_card': {
        'background': 'white',
        'borderRadius': '16px',
        'padding': '24px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.08)',
        'marginBottom': '16px',
        'borderLeft': f'5px solid {APP_CONFIG["theme_primary"]}',
        'transition': 'all 0.3s ease',
    },
    
    # Symptom Button
    'symptom_btn': {
        'background': 'linear-gradient(135deg, #ffffff 0%, #E8F5E9 100%)',
        'color': APP_CONFIG['theme_primary'],
        'border': f'2px solid {APP_CONFIG["theme_accent"]}',
        'padding': '12px 20px',
        'borderRadius': '25px',
        'fontSize': '14px',
        'fontWeight': '500',
        'cursor': 'pointer',
        'transition': 'all 0.3s ease',
        'margin': '5px',
    },
    
    # Modal Overlay
    'modal_overlay': {
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'right': '0',
        'bottom': '0',
        'backgroundColor': 'rgba(0,0,0,0.5)',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'zIndex': '1000',
    },
    
    # Modal Content
    'modal_content': {
        'background': 'white',
        'borderRadius': '20px',
        'padding': '40px',
        'maxWidth': '600px',
        'width': '90%',
        'maxHeight': '80vh',
        'overflow': 'auto',
        'boxShadow': '0 25px 50px rgba(0,0,0,0.25)',
    },
}


# =============================================================================
# 3. CUSTOM CSS (Animations & Advanced Styles)
# =============================================================================

CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Reset */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #E8F5E9;
    }
    ::-webkit-scrollbar-thumb {
        background: #00695C;
        border-radius: 5px;
    }
    
    /* Pulse Animation for Emergency Button */
    @keyframes pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 4px 15px rgba(211,47,47,0.4); }
        50% { transform: scale(1.02); box-shadow: 0 6px 25px rgba(211,47,47,0.6); }
    }
    
    /* Fade In Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Slide In Animation */
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Card Hover Effect */
    .medicine-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,105,92,0.15) !important;
    }
    
    /* Button Hover Effects */
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,105,92,0.4);
    }
    
    .symptom-btn:hover {
        background: linear-gradient(135deg, #00695C 0%, #004D40 100%) !important;
        color: white !important;
        transform: scale(1.05);
    }
    
    /* Search Input Focus */
    .search-input:focus {
        border-color: #00695C !important;
        box-shadow: 0 0 0 4px rgba(0,105,92,0.1), 0 4px 15px rgba(0,105,92,0.2) !important;
    }
    
    /* Loading Spinner */
    .loading-spinner {
        border: 4px solid #E8F5E9;
        border-top: 4px solid #00695C;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(135deg, #00695C 0%, #4DB6AC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Map Container */
    .map-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Match Score Badge */
    .match-badge {
        background: linear-gradient(135deg, #00695C 0%, #4DB6AC 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Severity Badges */
    .severity-low { background: #C8E6C9; color: #2E7D32; }
    .severity-moderate { background: #FFE0B2; color: #E65100; }
    .severity-high { background: #FFCDD2; color: #C62828; }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .search-container {
            padding: 20px !important;
        }
        .header-container {
            flex-direction: column;
            gap: 15px;
        }
    }
</style>
"""


# =============================================================================
# 4. LAYOUT COMPONENTS
# =============================================================================

def create_header(user_name='Guest'):
    """
    Create the application header with logo and user info
    
    Args:
        user_name: Current user's name
        
    Returns:
        Dash HTML component
    """
    return html.Header(
        style=STYLES['header'],
        children=[
            # Logo
            html.Div([
                html.Span('🏥', style={'fontSize': '36px'}),
                html.Span('CureBot', style=STYLES['logo']),
                html.Span('v3.0', style={
                    'fontSize': '12px',
                    'background': 'rgba(255,255,255,0.2)',
                    'padding': '4px 10px',
                    'borderRadius': '12px',
                    'marginLeft': '10px',
                }),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            
            # Right Side - User & Emergency
            html.Div([
                # Emergency Button
                html.Button([
                    html.Span('🚨', style={'marginRight': '8px'}),
                    'Emergency'
                ], id='emergency-btn', style=STYLES['btn_emergency']),
                
                # User Info
                html.Div([
                    html.Span(f'👤 {user_name}', style={
                        'color': 'white',
                        'fontSize': '16px',
                        'marginLeft': '20px',
                    })
                ]),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ]
    )


def create_search_section():
    """
    Create the main search section with input and symptom buttons
    
    Returns:
        Dash HTML component
    """
    # Common symptoms for quick selection
    common_symptoms = [
        '🤕 Headache', '🤒 Fever', '🤧 Cold', '😷 Cough',
        '💊 Pain', '🤢 Nausea', '😴 Insomnia', '🫁 Asthma',
        '❤️ Heart', '🦴 Joint Pain', '🧠 Anxiety', '🌡️ Diabetes'
    ]
    
    return html.Div(
        style={'padding': '40px', 'maxWidth': '1400px', 'margin': '0 auto'},
        children=[
            # Search Card
            html.Div(
                style=STYLES['card'],
                children=[
                    # Title
                    html.H2([
                        '🔍 Search for Medicines'
                    ], style={
                        'color': APP_CONFIG['theme_primary'],
                        'marginBottom': '25px',
                        'fontSize': '28px',
                        'fontWeight': '700',
                    }),
                    
                    # Search Input
                    html.Div([
                        dcc.Input(
                            id='symptom-input',
                            type='text',
                            placeholder='Enter your symptoms (e.g., headache, fever, cold)...',
                            style=STYLES['search_box'],
                            className='search-input',
                        ),
                    ], style={'marginBottom': '20px'}),
                    
                    # Search Button
                    html.Button([
                        html.Span('🔍', style={'marginRight': '10px'}),
                        'Search Medicines'
                    ], id='search-btn', style=STYLES['btn_primary'], className='btn-primary'),
                    
                    # Quick Symptom Buttons
                    html.Div([
                        html.P('Quick Select:', style={
                            'color': '#666',
                            'marginTop': '25px',
                            'marginBottom': '15px',
                            'fontWeight': '600',
                        }),
                        html.Div([
                            html.Button(
                                symptom,
                                id={'type': 'symptom-btn', 'index': i},
                                style=STYLES['symptom_btn'],
                                className='symptom-btn',
                                n_clicks=0,
                            ) for i, symptom in enumerate(common_symptoms)
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}),
                    ]),
                ]
            ),
        ]
    )


def create_results_section():
    """
    Create the results display section
    
    Returns:
        Dash HTML component with empty results container
    """
    return html.Div(
        id='results-container',
        style={'padding': '0 40px 40px', 'maxWidth': '1400px', 'margin': '0 auto'},
    )


def create_medicine_card(medicine, index):
    """
    Create a single medicine result card
    
    Args:
        medicine: Dictionary with medicine details
        index: Card index for animation delay
        
    Returns:
        Dash HTML component
    """
    return html.Div(
        style={
            **STYLES['medicine_card'],
            'animation': f'slideIn 0.5s ease {index * 0.1}s forwards',
            'opacity': '0',
        },
        className='medicine-card',
        children=[
            # Header Row
            html.Div([
                html.Div([
                    html.H3(medicine.get('Medicine Name', 'Unknown'), style={
                        'color': APP_CONFIG['theme_primary'],
                        'fontSize': '20px',
                        'fontWeight': '700',
                        'marginBottom': '5px',
                    }),
                    html.Span(medicine.get('Therapeutic Class', ''), style={
                        'color': '#666',
                        'fontSize': '14px',
                    }),
                ]),
                html.Span(medicine.get('Match Score', ''), style={
                    'background': f'linear-gradient(135deg, {APP_CONFIG["theme_primary"]} 0%, {APP_CONFIG["theme_accent"]} 100%)',
                    'color': 'white',
                    'padding': '8px 16px',
                    'borderRadius': '20px',
                    'fontWeight': '600',
                    'fontSize': '14px',
                }),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start', 'marginBottom': '15px'}),
            
            # Details
            html.Div([
                html.P([
                    html.Strong('Uses: '),
                    medicine.get('Uses', 'General medicine')
                ], style={'color': '#444', 'marginBottom': '8px', 'fontSize': '14px'}),
                html.P([
                    html.Strong('Side Effects: '),
                    medicine.get('Side Effects', 'Consult doctor')
                ], style={'color': '#666', 'marginBottom': '8px', 'fontSize': '14px'}),
                html.P([
                    html.Strong('Manufacturer: '),
                    medicine.get('Manufacturer', 'N/A')
                ], style={'color': '#888', 'fontSize': '13px'}),
            ]),
        ]
    )


def create_analytics_section(symptom):
    """
    Create the disease analytics section with graphs
    
    Args:
        symptom: User's symptom for analysis
        
    Returns:
        Dash HTML component with Plotly graph
    """
    return html.Div(
        style={**STYLES['card'], 'marginTop': '20px'},
        children=[
            html.H3([
                '📊 Disease Analytics: ',
                html.Span(symptom.title(), style={'color': APP_CONFIG['theme_accent']})
            ], style={
                'color': APP_CONFIG['theme_primary'],
                'marginBottom': '20px',
            }),
            dcc.Graph(
                id='analytics-graph',
                style={'height': '250px'},
            ),
        ]
    )


def create_map_section():
    """
    Create the OpenStreetMap section for nearby pharmacies/hospitals
    
    Returns:
        Dash HTML component with iframe map
    """
    return html.Div(
        style={**STYLES['card'], 'marginTop': '20px'},
        children=[
            html.H3([
                '🗺️ Find Nearby Pharmacies & Hospitals'
            ], style={
                'color': APP_CONFIG['theme_primary'],
                'marginBottom': '20px',
            }),
            html.Div([
                html.Iframe(
                    id='map-frame',
                    style={
                        'width': '100%',
                        'height': '400px',
                        'border': 'none',
                        'borderRadius': '12px',
                    },
                    srcDoc="""
                    <html>
                    <head>
                        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
                        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
                        <style>
                            body { margin: 0; }
                            #map { height: 100vh; width: 100%; }
                        </style>
                    </head>
                    <body>
                        <div id="map"></div>
                        <script>
                            var map = L.map('map').setView([20.5937, 78.9629], 5);
                            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                                attribution: '© OpenStreetMap'
                            }).addTo(map);
                            
                            if (navigator.geolocation) {
                                navigator.geolocation.getCurrentPosition(function(pos) {
                                    var lat = pos.coords.latitude;
                                    var lng = pos.coords.longitude;
                                    map.setView([lat, lng], 14);
                                    L.marker([lat, lng]).addTo(map).bindPopup('📍 Your Location').openPopup();
                                });
                            }
                        </script>
                    </body>
                    </html>
                    """
                ),
            ], className='map-container'),
        ]
    )


def create_emergency_modal():
    """
    Create the emergency mode modal
    
    Returns:
        Dash HTML component
    """
    return html.Div(
        id='emergency-modal',
        style={'display': 'none'},  # Hidden by default
        children=[
            html.Div(
                style=STYLES['modal_overlay'],
                children=[
                    html.Div(
                        style=STYLES['modal_content'],
                        children=[
                            html.Div([
                                html.Span('🚨', style={'fontSize': '48px'}),
                                html.H2('EMERGENCY MODE', style={
                                    'color': '#D32F2F',
                                    'marginLeft': '15px',
                                }),
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '25px'}),
                            
                            html.Div([
                                html.H4('Emergency Numbers:', style={'marginBottom': '15px'}),
                                html.P('🚑 Ambulance: 102', style={'fontSize': '18px', 'marginBottom': '10px'}),
                                html.P('🏥 Hospital: 104', style={'fontSize': '18px', 'marginBottom': '10px'}),
                                html.P('👮 Police: 100', style={'fontSize': '18px', 'marginBottom': '10px'}),
                                html.P('🔥 Fire: 101', style={'fontSize': '18px', 'marginBottom': '10px'}),
                            ], style={'marginBottom': '25px'}),
                            
                            html.Button('Close', id='close-emergency-btn', style={
                                'background': '#D32F2F',
                                'color': 'white',
                                'padding': '12px 30px',
                                'border': 'none',
                                'borderRadius': '8px',
                                'cursor': 'pointer',
                                'fontSize': '16px',
                            }),
                        ]
                    ),
                ]
            ),
        ]
    )


def create_login_page():
    """
    Create the Google Sign-In login page
    
    Returns:
        Dash HTML component
    """
    return html.Div(
        style={
            'minHeight': '100vh',
            'background': APP_CONFIG['theme_background'],
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center',
        },
        children=[
            html.Div(
                style={
                    'background': 'white',
                    'padding': '60px',
                    'borderRadius': '24px',
                    'boxShadow': '0 25px 50px rgba(0,0,0,0.15)',
                    'textAlign': 'center',
                    'maxWidth': '450px',
                },
                children=[
                    html.Div([
                        html.Span('🏥', style={'fontSize': '64px'}),
                    ]),
                    html.H1('CureBot', style={
                        'color': APP_CONFIG['theme_primary'],
                        'fontSize': '42px',
                        'fontWeight': '800',
                        'margin': '20px 0 10px',
                    }),
                    html.P('AI-Powered Medicine Recommendation', style={
                        'color': '#666',
                        'fontSize': '16px',
                        'marginBottom': '40px',
                    }),
                    
                    # Google Sign-In Button
                    html.A([
                        html.Img(
                            src='https://developers.google.com/identity/images/g-logo.png',
                            style={'width': '24px', 'marginRight': '12px'}
                        ),
                        'Sign in with Google'
                    ], 
                    id='google-signin-btn',
                    href='#',  # Will be handled by JavaScript
                    style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'background': 'white',
                        'border': '2px solid #E0E0E0',
                        'padding': '14px 30px',
                        'borderRadius': '12px',
                        'fontSize': '16px',
                        'fontWeight': '600',
                        'color': '#333',
                        'textDecoration': 'none',
                        'cursor': 'pointer',
                        'marginBottom': '20px',
                        'width': '100%',
                    }),
                    
                    # Continue as Guest
                    html.Button([
                        '👤 Continue as Guest'
                    ], id='guest-btn', style={
                        'background': 'transparent',
                        'border': f'2px solid {APP_CONFIG["theme_primary"]}',
                        'color': APP_CONFIG['theme_primary'],
                        'padding': '14px 30px',
                        'borderRadius': '12px',
                        'fontSize': '16px',
                        'fontWeight': '600',
                        'cursor': 'pointer',
                        'width': '100%',
                    }),
                ]
            ),
        ]
    )


def create_footer():
    """
    Create the application footer
    
    Returns:
        Dash HTML component
    """
    return html.Footer(
        style={
            'background': APP_CONFIG['theme_secondary'],
            'color': 'white',
            'padding': '30px 40px',
            'textAlign': 'center',
        },
        children=[
            html.P([
                '© 2025 CureBot v3.0 | Made with ❤️ by Team CureBot'
            ], style={'marginBottom': '10px'}),
            html.P([
                '⚠️ Disclaimer: This is for informational purposes only. Always consult a healthcare professional.'
            ], style={'fontSize': '12px', 'opacity': '0.8'}),
        ]
    )


# =============================================================================
# 5. GOOGLE OAUTH JAVASCRIPT
# =============================================================================

GOOGLE_OAUTH_SCRIPT = f"""
<script src="https://accounts.google.com/gsi/client" async defer></script>
<script>
    function handleCredentialResponse(response) {{
        // Decode JWT token
        const base64Url = response.credential.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {{
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }}).join(''));
        
        const user = JSON.parse(jsonPayload);
        
        // Store user info
        localStorage.setItem('curebotUser', JSON.stringify({{
            name: user.name,
            email: user.email,
            picture: user.picture,
            loggedIn: true
        }}));
        
        // Reload to update UI
        location.reload();
    }}
    
    window.onload = function() {{
        google.accounts.id.initialize({{
            client_id: '{APP_CONFIG["google_client_id"]}',
            callback: handleCredentialResponse
        }});
        
        google.accounts.id.renderButton(
            document.getElementById('google-signin-div'),
            {{ theme: 'outline', size: 'large', width: '100%' }}
        );
    }}
</script>
"""


# =============================================================================
# 6. LEAFLET MAP JAVASCRIPT
# =============================================================================

LEAFLET_MAP_JS = """
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>

<script>
function initMap() {
    // Initialize map centered on India
    var map = L.map('map').setView([20.5937, 78.9629], 5);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    // Get user location
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
            var lat = position.coords.latitude;
            var lon = position.coords.longitude;
            
            // Center map on user
            map.setView([lat, lon], 14);
            
            // Add user marker
            L.marker([lat, lon])
                .addTo(map)
                .bindPopup('📍 You are here!')
                .openPopup();
            
            // Search for nearby pharmacies using Overpass API
            searchNearby(lat, lon, map);
        });
    }
}

function searchNearby(lat, lon, map) {
    // Overpass API query for pharmacies and hospitals
    var query = `
        [out:json];
        (
            node["amenity"="pharmacy"](around:3000,${lat},${lon});
            node["amenity"="hospital"](around:5000,${lat},${lon});
        );
        out body;
    `;
    
    fetch('https://overpass-api.de/api/interpreter', {
        method: 'POST',
        body: query
    })
    .then(response => response.json())
    .then(data => {
        data.elements.forEach(place => {
            var icon = place.tags.amenity === 'pharmacy' ? '💊' : '🏥';
            var name = place.tags.name || place.tags.amenity;
            
            L.marker([place.lat, place.lon])
                .addTo(map)
                .bindPopup(`${icon} ${name}`);
        });
    });
}
</script>
"""


# =============================================================================
# 7. MAIN LAYOUT BUILDER
# =============================================================================

def build_main_layout():
    """
    Build the complete application layout
    
    Returns:
        Dash HTML component
    """
    return html.Div([
        # Custom CSS
        html.Div(
            dangerously_allow_html=True,
            children=CUSTOM_CSS,
        ) if hasattr(html.Div, 'dangerously_allow_html') else dcc.Markdown(''),
        
        # Main Container
        html.Div(
            style=STYLES['container'],
            children=[
                create_header(),
                create_search_section(),
                create_results_section(),
                create_footer(),
                create_emergency_modal(),
            ]
        ),
        
        # URL for page routing
        dcc.Location(id='url', refresh=False),
        
        # Hidden stores
        dcc.Store(id='user-store'),
        dcc.Store(id='search-results-store'),
    ])


# =============================================================================
# 8. DOCUMENTATION
# =============================================================================

"""
WEB UI COMPONENT DOCUMENTATION
==============================

LAYOUT STRUCTURE:
├── Header (Logo, User, Emergency)
├── Search Section
│   ├── Search Input
│   ├── Search Button
│   └── Quick Symptom Buttons
├── Results Section
│   ├── Medicine Cards
│   ├── Analytics Graphs
│   └── Map (OpenStreetMap)
├── Emergency Modal
└── Footer

STYLING APPROACH:
- Inline styles for component-level styling
- Custom CSS for animations and pseudo-elements
- CSS Variables for theme consistency
- Responsive design with media queries

COLOR PALETTE:
- Primary: #00695C (Teal)
- Secondary: #004D40 (Dark Teal)
- Accent: #4DB6AC (Light Teal)
- Background: Gradient (E8F5E9 → B2DFDB)
- Emergency: #D32F2F (Red)

COMPONENTS:
1. create_header() - Top navigation bar
2. create_search_section() - Main search interface
3. create_results_section() - Medicine results display
4. create_medicine_card() - Individual result cards
5. create_analytics_section() - Plotly graphs
6. create_map_section() - OpenStreetMap iframe
7. create_emergency_modal() - Emergency overlay
8. create_login_page() - Google OAuth login
9. create_footer() - Bottom section

EXTERNAL LIBRARIES:
- Leaflet.js (OpenStreetMap)
- Google Sign-In API
- Plotly.js (via Dash)
- Inter Font (Google Fonts)

CALLBACKS (defined in web.py):
- update_results: Handle search
- handle_symptom_click: Quick symptom buttons
- toggle_emergency: Emergency modal
- handle_login: Google/Guest login
"""

# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("   WEB UI MODULE - Testing Styles")
    print("=" * 60)
    
    print("\n✅ STYLES dictionary loaded with", len(STYLES), "style definitions")
    print("✅ APP_CONFIG loaded")
    print("✅ CUSTOM_CSS defined")
    print("✅ Component functions available:")
    print("   - create_header()")
    print("   - create_search_section()")
    print("   - create_results_section()")
    print("   - create_medicine_card()")
    print("   - create_analytics_section()")
    print("   - create_map_section()")
    print("   - create_emergency_modal()")
    print("   - create_login_page()")
    print("   - create_footer()")
    print("   - build_main_layout()")
    
    print("\n" + "=" * 60)
    print("   ✅ WEB UI MODULE TEST COMPLETE")
    print("=" * 60)
