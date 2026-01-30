# GitHub Repository Setup Guide
## Biometric Weight Estimation

**For**: Surya Yalavarthi | UC Computer Science | April 2026 Graduate

---

## 🚀 Quick Start for Recruiters

This repository contains a **research-grade machine learning project** that achieves **13.09 kg MAE** (3.04% better than published baseline) for body weight estimation from facial images.

### One-Command Demo

```bash
git clone https://github.com/YOUR_USERNAME/biometric-weight-estimation.git
cd biometric-weight-estimation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Demo runs at**: `http://localhost:8501`

---

## 📊 Repository Highlights

### Performance Metrics
- **MAE**: 13.09 kg (vs. 13.50 kg baseline) → **+3.04% improvement** ✅
- **Dataset**: 66,724 samples
- **Model Size**: 72 KB
- **Inference**: <1 second/image

### What's Included
- ✅ Complete source code (4 Python files)
- ✅ Trained model (`xgboost_weight_model.pkl` - 72 KB)
- ✅ Processed dataset (`facial_features_ratios_V2.csv` - 8.5 MB)
- ✅ SHAP visualizations (300 DPI)
- ✅ Production Streamlit app
- ✅ Comprehensive documentation

### What's NOT Included
- ❌ Raw image dataset (7 GB) - excluded via `.gitignore`
- ❌ Virtual environment (`.venv/`) - recreate with `pip install -r requirements.txt`

---

## 🎯 For Recruiters

**Key Skills Demonstrated**:
- Machine Learning (XGBoost, SHAP, scikit-learn)
- Data Engineering (ETL, memory optimization, 813:1 compression)
- Software Engineering (Streamlit, clean code, documentation)
- Research (beat published baseline, academic writing)

**Quantifiable Achievements**:
- Processed 66,866 images on 8GB RAM
- Achieved 813:1 data compression (7 GB → 8.5 MB)
- Deployed production web app with real-time predictions
- Generated publication-quality visualizations (300 DPI)

---

## 📁 Repository Structure

```
biometric-weight-estimation/
├── README.md                          # Technical overview
├── PORTFOLIO_SUMMARY.md               # Career-focused summary
├── RESEARCH_DISCUSSION.md             # Academic analysis
├── DEPLOYMENT_GUIDE.md                # Production deployment
│
├── extract_features_corrected.py      # Feature extraction
├── optimize_xgboost_with_shap.py      # Model training
├── save_model_for_deployment.py       # Model export
├── streamlit_app.py                   # Web application
│
├── xgboost_weight_model.pkl           # Trained model (72 KB)
├── idoc_weight_estimation/
│   └── facial_features_ratios_V2.csv  # Dataset (8.5 MB)
├── optimization_log.txt               # Training log
├── requirements.txt                   # Dependencies
│
├── shap_summary.png                   # Feature importance
├── shap_force_error_1.png             # Failure analysis
├── shap_force_error_2.png             # Failure analysis
└── shap_force_error_3.png             # Failure analysis
```

---

## 🔬 Reproducibility

All results are fully reproducible:

1. **Feature Extraction**: Run `extract_features_corrected.py` (or use provided CSV)
2. **Model Training**: Run `optimize_xgboost_with_shap.py`
3. **Deployment**: Run `streamlit run streamlit_app.py`

**Random Seed**: 42 (fixed for reproducibility)

---

## 📈 Release History

### v1.0 - Baseline Beaten (January 2026)
- ✅ Achieved 13.09 kg MAE (3.04% improvement)
- ✅ SHAP explainability analysis
- ✅ Production Streamlit deployment
- ✅ Complete documentation

---

## 🎓 Academic Context

**Course Relevance**:
- CS 6065 (Machine Learning)
- CS 6052 (Data Mining)
- CS 6053 (Computer Vision)
- CS 5001 (Software Engineering)

**Publication Ready**:
- Conference: ACM/IEEE Computer Vision
- Journal: Journal of Medical Systems
- Workshop: CVPR/ICCV Health Applications

---

## 📞 Contact

**Surya Yalavarthi**  
University of Cincinnati | Computer Science  
Expected Graduation: April 2026

[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [Email](mailto:YOUR_EMAIL)

---

## ⭐ Star This Repo

If you found this project useful or impressive, please consider starring it!

---

## 📜 License

Research and educational purposes only.  
Not licensed for commercial use without permission.
