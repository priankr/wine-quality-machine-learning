# Quick Start Guide 🚀

Get started with the Wine Quality Analysis in 5 minutes!

## Option 1: View Online (No Installation Required)

Visit **[Live Dashboard](https://priankr.github.io/wine-quality-machine-learning/)** to explore:
- Interactive visualizations
- Model performance comparisons
- Wine quality predictor
- Clustering analysis

## Option 2: Run Locally

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Launch Jupyter Notebook

```bash
jupyter notebook wine-quality-ml-updated.ipynb
```

### Step 3: Run All Cells

Click **Cell → Run All** or press **Shift + Enter** on each cell.

## What You'll See

### 1. Data Loading & EDA
- 6,497 wine samples loaded
- Interactive quality distribution charts
- Correlation heatmaps

### 2. Advanced ML Models
- **XGBoost** - ~70% accuracy
- **LightGBM** - ~72% accuracy (best!)
- **CatBoost** - ~71% accuracy
- Random Forest baseline - ~68% accuracy

### 3. SHAP Interpretability
- Feature importance rankings
- Which features drive quality predictions
- Individual prediction explanations

### 4. Clustering Analysis
- 4 natural wine clusters discovered
- PCA and UMAP visualizations
- Cluster characteristics

### 5. Exported Artifacts
All visualizations saved to `docs/` folder:
- `quality_distribution.html`
- `correlation_heatmap.html`
- `model_comparison.html`
- `pca_quality.html`
- `umap_clusters.html`
- And more!

## Expected Runtime

- **Full notebook:** ~5-10 minutes (depending on hardware)
- **Individual sections:** 30 seconds - 2 minutes each

## Troubleshooting

### ImportError: No module named 'xgboost'
```bash
pip install xgboost lightgbm catboost shap umap-learn
```

### Kernel Dies During Training
- Reduce `n_estimators` from 200 to 100
- Close other applications to free up memory

### Visualizations Don't Show
- Make sure `plotly` is installed: `pip install plotly`
- Restart Jupyter kernel: **Kernel → Restart**

## Next Steps

1. **Experiment:** Modify hyperparameters in the notebook
2. **Deploy:** Run the notebook to generate fresh visualizations
3. **Customize:** Edit `docs/index.html` to personalize the dashboard
4. **Share:** Push to GitHub and enable GitHub Pages

## Need Help?

- 📖 See [README.md](README.md) for full documentation
- 🐛 [Report issues](https://github.com/priankr/wine-quality-machine-learning/issues)
- 💬 Check Jupyter cell markdown for detailed explanations

Happy analyzing! 🍷
