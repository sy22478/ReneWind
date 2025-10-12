# 🌬️ ReneWind – Wind Turbine Failure Prediction

## 📌 Business Problem Statement
"ReneWind" is a company working on improving the machinery/processes involved in the production of wind energy using machine learning and has collected data of generator failure of wind turbines using sensors. They have shared a ciphered version of the data, as the data collected through sensors is confidential (the type of data collected varies with companies). Data has 40 predictors, 20000 observations in the training set and 5000 in the test set.

The objective is to build various classification models, tune them, and find the best one that will help identify failures so that generators could be repaired before failing/breaking to reduce maintenance costs. The nature of predictions made by the classification model will translate as follows:
- **True positives (TP)**: Maintenance happens when failure is going to occur → Good prediction
- **False positives (FP)**: Unnecessary maintenance → Acceptable (preventive maintenance)
- **True negatives (TN)**: No maintenance when no failure → Good prediction
- **False negatives (FN)**: No maintenance when failure occurs → **Catastrophic** (turbine damage, high replacement costs)

**Goal**: Minimize false negatives (maximize recall) to prevent catastrophic turbine failures while balancing precision to avoid excessive unnecessary maintenance.

---

## Project Overview
A comprehensive neural network-based predictive maintenance project for wind turbine generator failure detection. Uses TensorFlow/Keras to build 7 different neural network architectures for binary classification of wind turbine failures, with focus on handling class imbalance (17:1 ratio) and optimizing recall for critical failure detection.

## Complete Architecture

### ML System Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sensor Data   │    │  Data            │    │   ML Models     │
│   (40 Features) │───►│  Preprocessing   │───►│   Training      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                ┌──────▼──────┐                │
         │                │ Class       │                │
         │                │ Imbalance   │                │
         │                │ Handling    │                │
         │                └──────┬──────┘                │
         │                       │                       │
         │                ┌──────▼──────┐                │
         │                │ Threshold   │                │
         │                │ Tuning      │                │
         │                └─────────────┘                │
         │                                               ▼
┌─────────────────┐                               ┌───────────────┐
│   Evaluation    │◄──────────────────────────────│  Deployment   │
│   (Recall/FN)   │                               │  (Optional)   │
└─────────────────┘                               └───────────────┘
```

### Data Processing
- Standardization/Normalization of numeric features
- Handling class imbalance using class weights and/or oversampling (e.g., SMOTE)
- Train/validation split preserving class ratio

### Model Suite (7 Architectures)
1. Baseline Dense NN (2-3 layers)
2. Deep Dense NN with Dropout and BatchNorm
3. Residual MLP (skip connections)
4. Wide & Deep style MLP
5. MLP with L1/L2 regularization sweep
6. Focal-loss MLP (penalize FN more)
7. Calibrated MLP with temperature scaling

### Optimization & Tuning
- Learning rate schedules, early stopping
- Class-weighted loss vs focal loss comparison
- Threshold tuning to maximize recall at acceptable precision

## 📈 Evaluation
- Primary metric: Recall (minimize FN)
- Secondary metrics: Precision, F1, ROC-AUC, PR-AUC
- Confusion matrix analysis to quantify maintenance impact

## 🧪 Reproducibility
- Fixed seeds for NumPy/TensorFlow
- Logged hyperparameters and results for each architecture

## 🛠️ Tech Stack
- Python 3.10+
- TensorFlow/Keras, scikit-learn, imbalanced-learn
- NumPy, Pandas, Matplotlib/Seaborn

## 📦 Project Structure
```
ReneWind/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_models.ipynb
│   └── 04_threshold_tuning.ipynb
├── src/
│   ├── data.py
│   ├── models.py
│   ├── training.py
│   ├── metrics.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## ▶️ Usage
```bash
# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Train baseline model
python -m src.training --model baseline --epochs 50

# Run threshold tuning
python -m src.training --model deep --tune-threshold
```

## 🔮 Future Work
- Integrate SHAP for feature importance and model explainability
- Explore tree-based gradient boosting baselines for comparison
- Add real-time inference pipeline and alerting thresholds

## 👤 Author
- GitHub: @sy22478
- LinkedIn: https://www.linkedin.com/in/sonu-yadav-a61046245/
