# Kepler Telescope Data Analysis - Quick Start Guide

## 📋 Overview
This project provides comprehensive analysis of the Kepler telescope dataset including:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Machine learning models for exoplanet detection
- Professional visualizations

## 🚀 Quick Start

### Step 1: Install Required Packages

**Option A: Using the batch file (Recommended for Windows)**
```bash
install_packages.bat
```

**Option B: Manual installation**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter notebook ipykernel
```

**Option C: Using requirements.txt**
```bash
pip install -r requirements.txt
```

### Step 2: Run the Analysis

**Option A: Run the Python script**
```bash
python kepler_analysis.py
```
This will:
- Process the cumulative.csv file
- Generate 10 high-quality visualizations in the `visualizations/` folder
- Train 3 machine learning models
- Display comprehensive analysis results in the terminal

**Option B: Use the Jupyter Notebook (Recommended for interactive analysis)**
```bash
jupyter notebook kepler_analysis.ipynb
```
Then:
1. The notebook will open in your browser
2. Run all cells sequentially (Cell → Run All)
3. View inline visualizations and results
4. Export to PDF: File → Download as → PDF via LaTeX

## 📁 Project Files

### Core Files
- **cumulative.csv** - Kepler telescope dataset (raw data)
- **kepler_analysis.py** - Complete Python analysis script
- **kepler_analysis.ipynb** - Jupyter notebook version with detailed explanations

### Supporting Files
- **requirements.txt** - List of required Python packages
- **install_packages.bat** - Windows batch script for easy installation
- **README.md** - This file

### Output Files (Generated after running)
- **visualizations/** - Folder containing 10 professional plots:
  - `01_disposition_distribution.png` - Exoplanet classification breakdown
  - `02_correlation_heatmap.png` - Feature correlation matrix
  - `03_orbital_period_distribution.png` - Orbital period analysis
  - `04_planetary_radius_distribution.png` - Planet size distribution
  - `05_steff_vs_prad_scatter.png` - Stellar temp vs planet radius
  - `06_teq_by_disposition_boxplot.png` - Temperature by classification
  - `07_depth_vs_duration_scatter.png` - Transit characteristics
  - `08_model_comparison.png` - ML model performance metrics
  - `09_confusion_matrices.png` - Classification accuracy visualization
  - `10_feature_importance.png` - Most important predictive features

## 🔬 Analysis Components

### 1. Data Cleaning & Preprocessing
- Handles missing values using median imputation
- Removes duplicate records
- Outlier detection and capping using IQR method
- Feature engineering (creates derived features)
- Data standardization for machine learning

### 2. Exploratory Data Analysis (EDA)
- Statistical summaries of key features
- Distribution analysis of planetary and stellar characteristics
- Correlation analysis between features
- Comprehensive visualizations

### 3. Machine Learning Models
Three models are trained and compared:

**Logistic Regression** (Baseline)
- Fast, interpretable model
- Good for linear relationships

**Random Forest** (Ensemble method)
- Handles non-linear relationships
- Provides feature importance rankings
- Typically best performer

**Gradient Boosting** (Advanced ensemble)
- Sequential learning approach
- High accuracy potential
- Robust to overfitting

### 4. Key Metrics
All models are evaluated using:
- Accuracy - Overall correctness
- Precision - Positive prediction accuracy
- Recall - True positive detection rate
- F1-Score - Harmonic mean of precision and recall
- ROC-AUC - Model discrimination ability

## 📊 Expected Results

After running the analysis, you should see:
- **~9,000 astronomical observations** processed
- **~2,400 confirmed exoplanets** identified
- **Model accuracy >90%** for exoplanet classification
- **ROC-AUC score >0.95** for best model
- **Transit depth and stellar properties** as top predictive features

## 🎯 Key Findings

The analysis will reveal:
1. Distribution of confirmed vs. candidate exoplanets
2. Planetary radius ranges (mostly 1-20 Earth radii)
3. Orbital period patterns (0.5 to 500+ days)
4. Potentially habitable exoplanets (based on equilibrium temperature)
5. Most important features for detection

## 📝 Submission Format

For competition/assignment submission:

**Python Script**: Already created as `kepler_analysis.py`

**Jupyter Notebook**: Already created as `kepler_analysis.ipynb`

**PDF Export** (from Jupyter):
1. Open the notebook: `jupyter notebook kepler_analysis.ipynb`
2. Run all cells: Cell → Run All
3. Export to PDF: File → Download as → PDF via LaTeX
   - If LaTeX is not installed, use: File → Print Preview → Save as PDF

**Report Contents**:
- ✅ Data cleaning methodology
- ✅ Exploratory data analysis with visualizations
- ✅ Machine learning models and results
- ✅ Scientific insights and conclusions
- ✅ Well-commented code

## 🛠️ Troubleshooting

### "Module not found" errors
- Run `install_packages.bat` or install packages manually
- Verify Python is in your system PATH
- Try: `python -m pip install [package_name]`

### Script runs but no visualizations
- Check if `visualizations/` folder was created
- Verify matplotlib backend: `import matplotlib; print(matplotlib.get_backend())`
- Try adding `plt.show()` before `plt.savefig()` in the script

### Jupyter notebook won't open
- Verify installation: `pip list | findstr jupyter`
- Try: `python -m notebook kepler_analysis.ipynb`
- Alternative: Use Google Colab (upload the .ipynb file)

### Out of memory errors
- The dataset is large; ensure 4GB+ RAM available
- Close other applications
- Process data in chunks if needed

## 📚 Dataset Information

**Source**: NASA Kepler Mission - Cumulative Dataset

**Key Features**:
- `koi_disposition` - Classification (CONFIRMED, FALSE POSITIVE, CANDIDATE)
- `koi_period` - Orbital period (days)
- `koi_prad` - Planetary radius (Earth radii)
- `koi_teq` - Equilibrium temperature (Kelvin)
- `koi_depth` - Transit depth (ppm)
- `koi_steff` - Stellar effective temperature (K)
- And many more...

## 🏆 Project Highlights

✨ **Comprehensive Analysis**: Complete pipeline from raw data to insights
✨ **Professional Visualizations**: Publication-quality plots
✨ **Multiple ML Models**: Comparison of 3 different approaches
✨ **Scientific Rigor**: Statistical analysis and interpretation
✨ **Well-Documented**: Detailed comments and explanations
✨ **Reproducible**: Fixed random seeds for consistent results

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review error messages in the terminal
3. Verify all packages are installed correctly
4. Ensure cumulative.csv is in the same directory

---

**Good luck with your Stellar Analytics project! 🌟🔭**
