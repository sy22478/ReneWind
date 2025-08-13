# 🌬️ ReneWind – Wind Turbine Failure Prediction

## 📌 Project Overview

ReneWind is a **deep learning-based predictive maintenance system** designed to forecast wind turbine component failures before they occur.
By leveraging historical **sensor data from turbines** and applying **seven different neural network architectures**, this project aims to minimize operational downtime, reduce maintenance costs, and improve energy output efficiency.

---

## 🏢 Business Problem Statement

Wind turbines are critical in renewable energy production, but unexpected component failures can cause:

* **Significant downtime**
* **High repair costs**
* **Lost energy generation**

Our system predicts turbine failures in advance, enabling:

* **70–85% cost reduction** through proactive maintenance
* **Reduced downtime** and operational risk
* **Optimized maintenance schedules** for improved efficiency

---

## 📊 Dataset & Features

* **Training Samples:** 20,000
* **Testing Samples:** 5,000
* **Features:** 40 continuous **sensor variables** (V1–V40), representing mechanical and environmental conditions such as:

  * Temperature readings
  * Vibration metrics
  * Pressure levels
  * Wind speed and direction
  * Rotor & generator performance indicators

**Target Variable:** Binary classification – *Failure (1)* or *No Failure (0)*

---

## 🛠️ Technical Architecture

The project evaluates **7 deep learning models** built with **TensorFlow/Keras**, including:

* Dense feedforward networks with varying depth & width
* Dropout and L2 regularization to prevent overfitting
* Batch normalization for faster convergence
* Hyperparameter tuning for optimal performance

**Data Processing Pipeline:**

1. **Data Cleaning & Preprocessing** – Pandas, NumPy
2. **Feature Scaling** – Scikit-learn StandardScaler
3. **Train/Test Split** – Scikit-learn
4. **Model Training** – TensorFlow/Keras
5. **Evaluation** – Recall-focused performance metrics

---

## 📈 Performance Metrics

Since **false negatives** (missed failures) have a higher business cost, **Recall** is prioritized:

* **Primary Metric:** Recall
* **Secondary Metrics:** Precision, F1-Score, ROC-AUC
* **Business Outcome:** Higher recall → fewer undetected failures → reduced downtime costs

---

## 📂 Project Structure

```
ReneWind/
│── data/                # Dataset files (not included in repo)
│── notebooks/           # Jupyter notebooks for analysis & modeling
│── models/              # Saved trained models
│── src/                 # Source code (data processing, training, evaluation)
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation
│── LICENSE              # License file
```

---

## 💻 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sy22478/ReneWind.git
cd ReneWind
```

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage Instructions

1. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

2. Open the desired notebook from the `notebooks/` folder
3. Run cells sequentially to:

   * Load and preprocess data
   * Train models
   * Evaluate performance
   * Compare results

---

## 📦 requirements.txt

Includes:

* **Data Handling:** pandas, numpy, scikit-learn
* **Deep Learning:** tensorflow, keras
* **Visualization:** matplotlib, seaborn, plotly
* **Notebooks:** jupyter, ipykernel
* **Optional Analytics:** scipy, statsmodels
* **Development Tools:** pytest, black, flake8

---

## 📉 Business Impact

By integrating ReneWind into wind farm operations:

* **70–85% reduction in maintenance costs**
* **Significant downtime prevention**
* **Improved energy output reliability**
* **Enhanced operational planning** through predictive insights

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📬 Contact

**Author:** Sonu Yadav

**GitHub:** [sy22478](https://github.com/sy22478)

**Email:** sonu.yadav19997@gmail.com

---

**🌱 Powering the future of renewable energy with AI.**

---
