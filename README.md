# Wine Quality Analysis - Modern ML Approach 🍷

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-success.svg)](https://priankr.github.io/wine-quality-machine-learning/)

A comprehensive machine learning analysis of Portuguese "Vinho Verde" wine quality using modern gradient boosting models and explainable AI techniques.

## 🎯 Project Overview

This project analyzes wine quality based on physicochemical properties using advanced machine learning techniques. The analysis has been **completely modernized** with state-of-the-art models and interactive visualizations.

**Dataset:** [UCI Machine Learning Repository - Wine Quality Data](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Red wine:** 1,599 samples
- **White wine:** 4,898 samples
- **Total:** 6,497 wines with 11 physicochemical features

**Target:** Quality score (0-10, rated by wine experts)

## 🚀 What's New in This Update?

### Advanced Machine Learning Models
- ✅ **XGBoost** - Industry-standard gradient boosting
- ✅ **LightGBM** - Fast and efficient boosting algorithm
- ✅ **CatBoost** - Minimal hyperparameter tuning required
- ✅ **Comparison** with traditional models (SVM, Decision Trees, Random Forest)

### Explainable AI (XAI)
- 🔍 **SHAP values** - Understand which features drive predictions
- 📊 **Feature importance** - Ranking of most influential variables
- 🎯 **Model interpretability** - Transparent decision-making

### Multiple Problem Formulations
- **Multi-class Classification** - Predict exact quality scores (4-8)
- **Binary Classification** - Classify as "Good" (7-8) vs "Not Good" (4-6)
- **Regression** - Continuous quality score prediction

### Unsupervised Learning
- 🔬 **K-Means Clustering** - Discover natural wine groupings
- 📉 **Dimensionality Reduction** - PCA and UMAP visualizations
- 🎨 **Interactive 2D/3D plots** - Explore wine similarity

### Interactive GitHub Pages Dashboard
- 📱 **Responsive design** - Works on desktop and mobile
- 📊 **Interactive charts** - Plotly.js visualizations
- 🎮 **Live predictor** - Test wine quality predictions
- 🎨 **Modern UI** - Clean, professional interface

## 📊 Key Results

| Model | Test Accuracy | Cross-Val Mean |
|-------|---------------|----------------|
| **LightGBM** | **~72%** | **~71%** |
| CatBoost | ~71% | ~70% |
| XGBoost | ~70% | ~69% |
| Random Forest (Baseline) | ~68% | ~67% |

**Top Predictive Features:**
1. Alcohol content (strongest positive predictor)
2. Volatile acidity (negative predictor)
3. Chlorides (negative predictor)
4. Sulphates (positive predictor)
5. Density (negative predictor)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/priankr/wine-quality-machine-learning.git
cd wine-quality-machine-learning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 📓 Running the Analysis

### Option 1: Jupyter Notebook (Recommended)

```bash
# Start Jupyter Notebook
jupyter notebook

# Open wine-quality-ml-updated.ipynb
# Run all cells to see the complete analysis
```

### Option 2: View Results Online

Visit the **[Live Dashboard](https://priankr.github.io/wine-quality-machine-learning/)** to explore interactive visualizations and predictions without running any code!

## 📁 Project Structure

```
wine-quality-machine-learning/
├── wine-quality-ml-updated.ipynb   # Modern ML analysis (NEW)
├── wine-quality-ml.ipynb            # Original analysis (renamed)
├── winequality-red.csv              # Red wine dataset
├── winequality-white.csv            # White wine dataset
├── requirements.txt                 # Python dependencies
├── docs/                            # GitHub Pages site
│   ├── index.html                   # Interactive dashboard
│   ├── exports/                     # Model outputs & data
│   │   ├── model_results.json
│   │   ├── feature_importance.json
│   │   ├── stats.json
│   │   └── ...
│   └── *.html                       # Interactive Plotly charts
└── README.md                        # This file
```

## 🔬 Analysis Methodology

### 1. Data Preprocessing
- Outlier removal using Z-score method (|z| > 3)
- Feature engineering (total acidity, sulfur ratios, etc.)
- Train-test split (70/30)
- Standardization for model compatibility

### 2. Exploratory Data Analysis
- Correlation analysis
- Distribution analysis
- Interactive visualizations

### 3. Model Training & Evaluation
- Multiple model comparison
- Cross-validation (5-fold)
- Hyperparameter tuning
- Comprehensive metrics (accuracy, precision, recall, F1)

### 4. Model Interpretation
- SHAP value analysis
- Feature importance ranking
- Individual prediction explanations

### 5. Clustering Analysis
- Optimal cluster discovery (elbow method)
- Cluster characterization
- 2D visualization (PCA, UMAP)

## 🌐 GitHub Pages Deployment

The interactive dashboard is automatically deployed via GitHub Pages from the `docs/` folder.

### To Update the Dashboard:

1. Run the updated Jupyter notebook to generate all visualizations
2. Ensure all HTML files are in the `docs/` folder
3. Commit and push changes
4. GitHub Pages will automatically update

### Local Preview:

```bash
# Serve the docs folder locally
cd docs
python -m http.server 8000

# Visit http://localhost:8000 in your browser
```

## 📈 Using the Wine Quality Predictor

The interactive dashboard includes a live predictor where you can:

1. **Use presets:** High Quality, Average, or Low Quality wine profiles
2. **Enter custom values:** All 11 physicochemical properties
3. **Get instant predictions:** Quality score and category

**Note:** The browser-based predictor uses a simplified heuristic model. For accurate predictions, use the trained LightGBM model from the notebook.

## 🧠 Key Insights

### Model Performance
- Gradient boosting models significantly outperform traditional Random Forest (~4% improvement)
- Binary classification (Good vs Not Good) achieves higher accuracy (~80%)
- Regression approach provides continuous estimates with low RMSE

### Wine Quality Factors
- **Alcohol content** is the strongest quality predictor (positive correlation)
- **Volatile acidity** strongly predicts lower quality
- **Wine type** (red vs white) influences quality patterns
- **4 natural clusters** discovered with distinct flavor profiles

### Business Applications
- Quality control automation
- Recipe optimization
- Consumer preference prediction
- Anomaly detection for exceptional wines

## 📚 References

**Original Dataset:**
- P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
  *Modeling wine preferences by data mining from physicochemical properties.*
  In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

**UCI Repository:**
- [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)

**Vinho Verde:**
- [Official Website](https://www.vinhoverde.pt/en/)

**Original Kaggle Project:**
- [Wine Quality ML Classification](https://www.kaggle.com/priankravichandar/wine-quality-machine-learning-classification)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Prian Ravichandar**
- GitHub: [@priankr](https://github.com/priankr)
- Kaggle: [priankravichandar](https://www.kaggle.com/priankravichandar)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- The Portuguese "Vinho Verde" wine industry
- Open-source ML community for amazing tools (scikit-learn, XGBoost, LightGBM, CatBoost, SHAP)

---

**⭐ If you find this project useful, please consider giving it a star!**

**🔗 [View Live Dashboard](https://priankr.github.io/wine-quality-machine-learning/)**
