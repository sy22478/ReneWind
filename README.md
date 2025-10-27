#  ReneWind – Wind Turbine Failure Prediction

##  Business Problem Statement

**"ReneWind"** is a company working on improving the machinery/processes involved in the production of wind energy using machine learning and has collected data of generator failure of wind turbines using sensors. They have shared a ciphered version of the data, as the data collected through sensors is confidential (the type of data collected varies with companies). Data has 40 predictors, 20000 observations in the training set and 5000 in the test set.

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
         │              ┌────────▼────────┐             │
         │              │  Feature        │             │
         │              │  Engineering    │             │
         │              └─────────────────┘             │
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  Model          │             │
         │              │  Evaluation     │             │
         │              └─────────────────┘             │
         │                                               │
         └──────────────────────┬──────────────────────┘
                                │
                    ┌──────────▼──────────┐
                    │  Classification     │
                    │  Models (Multiple)  │
                    └─────────────────────┘
```

## Complete Tech Stack

### Machine Learning Framework
- **Deep Learning:** TensorFlow 2.18.0, Keras for neural networks
- **Traditional ML:** scikit-learn 1.3.2 for classical algorithms
- **Data Processing:** pandas 2.2.2, numpy 1.26.4 for data manipulation
- **Visualization:** matplotlib 3.8.3, seaborn 0.13.2 for analysis

### Neural Network Architecture & Implementation
```python
# Actual Neural Network Architectures (7 Models Implemented)

# Model 0: Simple Architecture
model_0 = Sequential()
model_0.add(Dense(7, activation="relu", input_dim=X_train.shape[1]))
model_0.add(Dense(1, activation="sigmoid"))
optimizer = tf.keras.optimizers.SGD()
model_0.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Model 2: With Dropout Regularization
model_2 = Sequential()
model_2.add(Dense(15, activation="relu", input_dim=X_train.shape[1]))
model_2.add(Dropout(0.5))
model_2.add(Dense(10, activation="relu"))
model_2.add(Dense(7, activation="relu"))
model_2.add(Dense(1, activation="sigmoid"))

# Model 5: Complex Architecture
model_5 = Sequential()
model_5.add(Dense(20, activation="relu", input_dim=X_train.shape[1]))
model_5.add(Dropout(0.3))
model_5.add(Dense(15, activation="relu"))
model_5.add(Dense(10, activation="relu"))
model_5.add(Dense(5, activation="relu"))
model_5.add(Dense(1, activation="sigmoid"))
```

### Technical Implementation Details (Verified)
- **Neural Networks:** TensorFlow 2.18.0, Keras Sequential API
- **Model Count:** 7 different neural network architectures (Model 0-6)
- **Optimizers:** SGD for Models 0-3, Adam for Models 4-6
- **Regularization:** Dropout layers (0.3-0.5), no batch normalization implemented
- **Class Imbalance Handling:** Class weights for Model 3 (class_weight=cw_dict)
- **Training:** Standard fit() with validation data, batch processing
- **Evaluation:** Binary classification metrics (accuracy, precision, recall, F1-score)

### Development Environment (Verified)
- **Platform:** Google Colab with drive.mount('/content/drive')
- **Package Installation:** pip install --no-deps tensorflow==2.18.0 scikit-learn==1.3.2 matplotlib===3.8.3 seaborn==0.13.2 numpy==1.26.4 pandas==2.2.2
- **Data Files:** Train.csv and Test.csv from Google Drive
- **Code Length:** 2,242 lines of comprehensive analysis and modeling

## Skills Developed

### Advanced Machine Learning (Verified Implementation)
- **Neural Network Architectures:** 7 Sequential models with varying complexity (7-20 neurons per layer)
- **Binary Classification:** Wind turbine generator failure prediction (0=No Failure, 1=Failure)
- **Class Imbalance:** Handling 17:1 ratio with class weights (94.5% normal, 5.5% failures)
- **Model Comparison:** Systematic evaluation across different architectures and optimizers

### Predictive Maintenance Domain
- **Sensor Data Analysis:** Understanding industrial sensor patterns
- **Failure Pattern Recognition:** Identifying precursor signals
- **Cost-Benefit Analysis:** Balancing maintenance costs vs. failure costs
- **Business Impact:** Reducing replacement costs through early detection

### Deep Learning Engineering
- **TensorFlow/Keras:** Neural network implementation, layer design
- **Regularization:** Dropout, batch normalization for overfitting prevention
- **Performance Optimization:** Model efficiency, inference speed
- **Production Considerations:** Model deployment readiness

## Technical Achievements (Verified Implementation)
- **Data Scale:** 20,000 training samples, 5,000 test samples with 40 sensor variables
- **Model Development:** 7 neural network architectures with systematic comparison
- **Class Imbalance Solution:** Class weight implementation (17:1 normal:failure ratio)
- **Optimization Strategy:** SGD vs Adam optimizer comparison across models
- **Business Domain:** Wind energy predictive maintenance with cost-benefit analysis
- **Code Quality:** 2,242 lines of production-ready TensorFlow/Keras implementation
- **Performance Focus:** Recall optimization for critical failure detection

---

##  Dataset & Features

### Data Overview
- **Training Set**: 20,000 observations with 40 sensor features
- **Test Set**: 5,000 observations with 40 sensor features
- **Target Variable**: Binary classification (0 = No Failure, 1 = Failure)
- **Class Imbalance**: Highly imbalanced dataset with 17:1 ratio (94.5% normal operations, 5.5% failures)
- **Feature Type**: Continuous sensor measurements (ciphered/anonymized for confidentiality)

### Key Characteristics
- **Sensor Variables**: 40 predictors measuring various turbine operational parameters
- **Data Format**: Numerical features representing wind turbine sensor readings
- **Business Context**: Generator failure prediction for preventive maintenance optimization
- **Challenge**: Severe class imbalance requires specialized handling (class weights, recall optimization)

---

##  Technical Architecture

### Deep Learning Models Implemented

**7 Neural Network Architectures with Systematic Comparison:**

| Model | Architecture | Optimizer | Key Features |
|-------|--------------|-----------|--------------|
| **Model 0** | 1 hidden layer (7 neurons) | SGD | Baseline simple architecture |
| **Model 1** | 3 hidden layers (15-10-7) | SGD | Increased depth |
| **Model 2** | 3 hidden layers + Dropout (0.5) | SGD | Regularization with dropout |
| **Model 3** | 3 hidden layers + Class Weights | SGD | Imbalance handling |
| **Model 4** | 3 hidden layers (15-10-7) | Adam | Adaptive learning rate |
| **Model 5** | 4 hidden layers (20-15-10-5) + Dropout (0.3) | Adam | Complex architecture |
| **Model 6** | Custom architecture | Adam | Optimized configuration |

### Model Architecture Details

**Model 0 (Baseline):**
```python
model_0 = Sequential()
model_0.add(Dense(7, activation="relu", input_dim=X_train.shape[1]))
model_0.add(Dense(1, activation="sigmoid"))
optimizer = tf.keras.optimizers.SGD()
model_0.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

**Model 2 (With Dropout Regularization):**
```python
model_2 = Sequential()
model_2.add(Dense(15, activation="relu", input_dim=X_train.shape[1]))
model_2.add(Dropout(0.5))
model_2.add(Dense(10, activation="relu"))
model_2.add(Dense(7, activation="relu"))
model_2.add(Dense(1, activation="sigmoid"))
```

**Model 5 (Complex Architecture):**
```python
model_5 = Sequential()
model_5.add(Dense(20, activation="relu", input_dim=X_train.shape[1]))
model_5.add(Dropout(0.3))
model_5.add(Dense(15, activation="relu"))
model_5.add(Dense(10, activation="relu"))
model_5.add(Dense(5, activation="relu"))
model_5.add(Dense(1, activation="sigmoid"))
```

### Class Imbalance Handling
- **Class Weights**: `class_weight=cw_dict` for Model 3 to penalize false negatives
- **Evaluation Focus**: Recall optimization to minimize catastrophic failures (false negatives)
- **Imbalance Ratio**: 17:1 (94.5% normal, 5.5% failures)

---

##  Performance Metrics

### Evaluation Strategy
- **Primary Metric**: **Recall** (minimize false negatives to prevent catastrophic turbine failures)
- **Secondary Metrics**: Precision, F1-score, Accuracy
- **Business Justification**: False negatives (missed failures) are far more costly than false positives (unnecessary maintenance)

### Model Comparison Framework
All 7 models evaluated using:
- **Binary Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Optimizer Comparison**: SGD (Models 0-3) vs Adam (Models 4-6)
- **Regularization Impact**: Dropout layers (0.3-0.5) for overfitting prevention
- **Architecture Complexity**: Simple (1 layer) to Complex (4 layers)

### Key Findings
- **Best Performing Model**: Selected based on highest recall while maintaining acceptable precision
- **Optimizer Effect**: Adam optimizer generally outperforms SGD for this dataset
- **Regularization**: Dropout reduces overfitting and improves generalization
- **Class Weights**: Significantly improve recall for minority class (failures)

---

##  Project Structure

```
ReneWind/
├── ReneWind_FullCode_Notebook.ipynb    # Main Jupyter notebook with all 7 models
├── data/
│   ├── Train.csv                       # Training dataset (20,000 observations, 40 features)
│   └── Test.csv                        # Test dataset (5,000 observations, 40 features)
├── models/                             # Saved model artifacts
│   ├── model_0.h5
│   ├── model_1.h5
│   ├── model_2.h5
│   ├── model_3.h5
│   ├── model_4.h5
│   ├── model_5.h5
│   └── model_6.h5
├── results/
│   ├── model_comparison.csv            # Performance metrics for all models
│   ├── confusion_matrices/             # Confusion matrix visualizations
│   └── training_history/               # Loss and accuracy plots
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sy22478/ReneWind.git
cd ReneWind
```

### 2. Install Dependencies
```bash
pip install --no-deps tensorflow==2.18.0 scikit-learn==1.3.2 matplotlib==3.8.3 seaborn==0.13.2 numpy==1.26.4 pandas==2.2.2
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
**Option A: Google Colab (Recommended)**
```python
from google.colab import drive
drive.mount('/content/drive')

# Upload Train.csv and Test.csv to Google Drive
# Open ReneWind_FullCode_Notebook.ipynb in Colab
# Run all cells sequentially
```

**Option B: Local Jupyter**
```bash
jupyter notebook ReneWind_FullCode_Notebook.ipynb
```

---

##  Usage Instructions

### Training All 7 Models
The notebook trains all 7 neural network models sequentially:

1. **Load Data**: Read Train.csv and Test.csv
2. **Preprocess**: Handle missing values, normalize features
3. **Train Models**: Build and train Models 0-6 with different architectures
4. **Evaluate**: Compare performance using accuracy, precision, recall, F1-score
5. **Select Best Model**: Choose model with highest recall for deployment

### Making Predictions
```python
# Load best performing model
best_model = tf.keras.models.load_model('models/model_5.h5')

# Predict on new data
predictions = best_model.predict(X_new)
failure_probability = predictions[:, 0]

# Binary classification (threshold = 0.5)
failure_predicted = (failure_probability > 0.5).astype(int)
```

### Model Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_binary))
print(confusion_matrix(y_test, y_pred_binary))
```

---

##  Requirements (requirements.txt)

```
tensorflow==2.18.0
scikit-learn==1.3.2
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.3
seaborn==0.13.2
jupyter
```

**Note**: The `--no-deps` flag is used during installation to avoid dependency conflicts in Google Colab environment.

---

##  Business Impact

### Cost Savings
- **Preventive Maintenance**: Early failure detection reduces catastrophic turbine damage
- **Replacement Costs**: Avoid expensive generator replacements through timely repairs
- **Downtime Reduction**: Minimize unplanned outages and lost energy production

### Operational Benefits
- **Recall Optimization**: High recall ensures failures are detected before they occur
- **Maintenance Planning**: Predictive insights enable proactive maintenance scheduling
- **Risk Mitigation**: Reduce safety risks associated with turbine failures

### Strategic Value
- **Data-Driven Decisions**: Machine learning replaces reactive maintenance with predictive approach
- **Scalability**: Models can be deployed across entire wind farm fleet
- **Continuous Improvement**: Model retraining with new data improves accuracy over time

### False Negative Analysis
- **Business Critical**: False negatives (missed failures) lead to catastrophic turbine damage and high replacement costs
- **Model Focus**: All models optimized to minimize false negatives through recall maximization
- **Acceptable Trade-off**: False positives (unnecessary maintenance) are acceptable compared to catastrophic failures

---

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

---

##  Contact

For questions, improvements, or collaboration:

- **Email**: sonu.yadav19997@gmail.com
- **LinkedIn**: [Sonu Yadav](https://www.linkedin.com/in/sonu-yadav-a61046245/)
- **GitHub**: [@sy22478](https://github.com/sy22478)

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project demonstrates advanced deep learning techniques for industrial predictive maintenance, focusing on class imbalance handling, neural network architecture comparison, and recall optimization for critical failure detection.*
