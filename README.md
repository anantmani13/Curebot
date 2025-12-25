# 🏥 CureBot - AI-Powered Medicine Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Dash-3.x-00695C?style=for-the-badge&logo=plotly" alt="Dash">
  <img src="https://img.shields.io/badge/ML-TF--IDF-orange?style=for-the-badge&logo=scikit-learn" alt="ML">
  <img src="https://img.shields.io/badge/Google-Cloud-4285F4?style=for-the-badge&logo=google-cloud" alt="Google Cloud">
  <img src="https://img.shields.io/badge/Gemini-AI-8E75B2?style=for-the-badge&logo=google" alt="Gemini AI">
  <img src="https://img.shields.io/badge/Maps-OpenStreetMap-green?style=for-the-badge&logo=openstreetmap" alt="OpenStreetMap">
</p>

<p align="center">
  <b>🚀 Built for Google Developers Group (GDG) | Powered by Google Technologies</b>
</p>

---

## 🌟 Google Technologies Used

| Technology | Usage |
|------------|-------|
| 🔐 **Google Sign-In (OAuth 2.0)** | Secure user authentication |
| 🤖 **Google Gemini AI** | Smart health advice and recommendations |
| 📊 **Google AI Studio** | AI model integration |
| 🗺️ **Google Maps Directions** | Navigation to pharmacies/hospitals |
| ☁️ **Google Cloud Platform** | API hosting and management |

---

## ✨ Features

### Core Features
| Feature | Description |
|---------|-------------|
| 🤖 **AI-Powered Recommendations** | TF-IDF + Cosine Similarity ML model with n-grams (1-4) |
| 💊 **248,000+ Medicines** | Comprehensive database with uses, forms, and side effects |
| 🧠 **30 Symptom Categories** | Smart synonym matching for accurate results |
| 📊 **Interactive Disease Analytics** | Plotly graphs showing recovery rates, prevalence, severity |
| 🔐 **Google Sign-In** | Secure OAuth 2.0 authentication |
| 💎 **Google Gemini AI** | AI-powered health advice integration |

### Map and Location Features
| Feature | Description |
|---------|-------------|
| 🗺️ **Find Nearby Pharmacies** | FREE OpenStreetMap + Leaflet.js integration |
| 🏥 **Hospital Locator** | Emergency mode finds nearest hospitals |
| 📞 **Contact Info** | Phone numbers and directions via Google Maps |
| 📍 **Real-time Location** | GPS-based nearest location finding |

### Emergency Mode 🚨
| Feature | Description |
|---------|-------------|
| 🔴 **Emergency UI** | Full-screen red danger interface |
| 🏥 **Nearest Hospitals** | Auto-detects hospitals within 10km |
| 📞 **Quick Dial** | 102, 108, 112, 100 emergency numbers |
| 🚑 **Directions** | One-click navigation to hospitals |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anantmani13/Curebot.git
   cd Curebot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Google APIs** (Optional but recommended)
   
   Edit `web.py` and add your API keys:
   ```python
   GOOGLE_CLIENT_ID = "your-client-id.apps.googleusercontent.com"
   GEMINI_API_KEY = "your-gemini-api-key"
   ```
   
   Get keys from:
   - Google Sign-In: https://console.cloud.google.com/
   - Gemini API: https://aistudio.google.com/

5. **Run the application**
   ```bash
   python web.py
   ```

6. **Open in browser**
   ```
   http://127.0.0.1:8051
   ```

---

## 🔧 API Configuration

### Google Sign-In Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "Google Sign-In API"
4. Create OAuth 2.0 credentials
5. Add authorized JavaScript origins: `http://localhost:8051`
6. Copy Client ID to `web.py`

### Google Gemini AI Setup
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new API key
3. Copy the key to `web.py`

---

## 🧠 Machine Learning Model

- **Algorithm**: TF-IDF Vectorizer + Cosine Similarity
- **N-grams**: 1-4 (unigrams to 4-grams)
- **Max Features**: 15,000
- **Sublinear TF**: Enabled
- **Stop Words**: English
- **L2 Normalization**: Applied
- **Symptom Categories**: 30 synonym groups

---

## 📊 Disease Analytics

Interactive Plotly graphs showing:
- **Recovery Rate** - Gauge chart showing success rate
- **Prevalence** - How common the condition is
- **Average Duration** - Days to recovery
- **Severity Level** - Low / Medium / High

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

**CureBot is for educational purposes only.** It is NOT a substitute for professional medical advice. Always consult a qualified healthcare provider.

**In case of emergency, call:**
- 🚑 Ambulance: **102** / **108**
- 🆘 Emergency: **112**
- 👮 Police: **100**

---

## 👨‍💻 Team

Built with ❤️ for **Google Developers Group (GDG)**

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-blue?style=flat-square&logo=python" alt="Made with Python">
  <img src="https://img.shields.io/badge/Powered%20by-Google%20AI-4285F4?style=flat-square&logo=google" alt="Powered by Google AI">
</p>
