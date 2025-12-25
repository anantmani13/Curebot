# 🏥 CureBot - AI-Powered Medicine Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Dash-3.x-00695C?style=for-the-badge&logo=plotly" alt="Dash">
  <img src="https://img.shields.io/badge/ML-TF--IDF-orange?style=for-the-badge&logo=scikit-learn" alt="ML">
  <img src="https://img.shields.io/badge/Maps-OpenStreetMap-green?style=for-the-badge&logo=openstreetmap" alt="OpenStreetMap">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<p align="center">
  <b>An intelligent medicine recommendation chatbot that suggests medicines based on symptoms using Machine Learning.</b>
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI-Powered Recommendations** | Uses TF-IDF + Cosine Similarity to match symptoms with medicines |
| 💊 **248,000+ Medicines** | Comprehensive database of medicines with uses and side effects |
| 🗺️ **Find Nearby Pharmacies** | FREE OpenStreetMap integration to locate pharmacies near you |
| ⚡ **Quick Symptom Buttons** | 14 pre-configured symptom shortcuts for fast searching |
| 🎨 **Premium UI** | Beautiful hospital-green themed interface with smooth animations |
| 🚨 **Emergency Detection** | Automatically detects emergency keywords and shows helpline numbers |
| 📱 **Responsive Design** | Works on desktop and mobile devices |

---

## 🖥️ Screenshots

### Main Interface
```
┌──────────────────────────────────────────────────────────────┐
│  ➕ CureBot - Your AI-Powered Medicine Assistant             │
│  ────────────────────────────────────────────────────────────│
│  🟢 AI Active • 248,218 medicines    📍 Find Nearby Pharmacy │
├──────────────────────────────────────────────────────────────┤
│  🤖 Welcome to CureBot!                                      │
│     I'm your AI medicine assistant. Describe your symptoms   │
│     and I'll help you find the right medicine.               │
│                                                              │
│  ⚡ Quick Search:                                            │
│  [🤕 Headache] [🌡️ Fever] [🤧 Cold] [😷 Cough] [💪 Pain]    │
│  [🤢 Nausea] [😴 Sleep] [🤧 Allergy] [🩸 Diabetes] ...      │
│                                                              │
│  💬 [Describe your symptoms...              ] [🔍 Find]      │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anantmani13/CureBot.git
   cd CureBot
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
   pip install dash pandas numpy scikit-learn
   ```

4. **Download datasets** (Required)
   - `all_medicine databased.csv` (248,218 medicines)
   - `medicine_dataset.csv` (50,000 inventory items)
   
   > Place both CSV files in the same directory as `web.py`

5. **Run the application**
   ```bash
   python web.py
   ```

6. **Open in browser**
   ```
   http://127.0.0.1:8051
   ```

---

## 📁 Project Structure

```
CureBot/
├── web.py                      # Main application (Dash + ML)
├── medimatch_app.py            # Backup/alternative ML script
├── all_medicine databased.csv  # Primary medicine database
├── medicine_dataset.csv        # Secondary inventory data
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## 🧠 How It Works

### Machine Learning Pipeline

```
User Input (Symptoms)
        │
        ▼
┌───────────────────┐
│ Symptom Expansion │  ← 30 symptom synonym categories
│   (Preprocessing) │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  TF-IDF Vectorizer │  ← N-grams (1-4), 15,000 features
│                    │
└─────────┬──────────┘
          │
          ▼
┌───────────────────┐
│ Cosine Similarity │  ← Match against 248K medicines
│                   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Top 12 Results   │  ← Ranked by relevance score
└───────────────────┘
```

### Key Technologies

| Component | Technology |
|-----------|------------|
| Frontend | Dash (Python) |
| ML Model | TF-IDF + Cosine Similarity |
| Maps | OpenStreetMap + Leaflet.js (FREE!) |
| Styling | Custom CSS with Glassmorphism |

---

## 🗺️ Pharmacy Finder

CureBot includes a **FREE** pharmacy locator powered by OpenStreetMap:

- Click **"📍 Find Nearby Pharmacy"**
- Allow location access
- View nearby pharmacies on an interactive map
- Get directions to any pharmacy

**No API key required!** 🆓

---

## ⚠️ Disclaimer

> **CureBot is for educational purposes only.**
> 
> - This is NOT a substitute for professional medical advice
> - Always consult a qualified doctor before taking any medication
> - In case of emergency, call: **102 / 108 / 911**

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **Dash** - Web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine Learning
- **Leaflet.js** - Interactive maps
- **OpenStreetMap** - Free map tiles

---

## 📊 Dataset Information

| Dataset | Records | Size |
|---------|---------|------|
| Medicine Database | 248,218 | ~85 MB |
| Inventory Data | 50,000 | ~3.7 MB |

**Columns include:**
- Medicine Name
- Primary Use
- Therapeutic Class
- Side Effects (0-10)
- Dosage Form
- Classification (OTC/Prescription)

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Anant Mani**
- GitHub: [@anantmani13](https://github.com/anantmani13)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

<p align="center">
  Made with ❤️ for Google Developer Group
</p>
