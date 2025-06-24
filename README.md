# 🏠 House Price Predictor with Linear Regression

**House Price Predictor** is a Python-based end-to-end machine learning project that predicts residential house prices using a linear regression model. It follows best practices with modular code structure, reproducible preprocessing, and clear evaluation metrics.

---

## 🚀 Features

- **Data Preprocessing**: Automatically handles missing values and categorical encoding.
- **Linear Regression Model**: Implements a simple, interpretable machine learning algorithm to predict house prices.
- **Model Training & Evaluation**: Separates training and testing logic with consistent evaluation metrics.
- **Predictions from CSV**: Predicts prices for new data using a separate input CSV file.
- **Modular Pipeline**: Organized project structure with `main.py`, `train.py`, `evaluate.py`, `predict.py`, and `utils.py`.

---

## 🛠️ Tech Stack & Libraries Used

- **Python 3.x**
- **Libraries**:
  - **Pandas**: For data manipulation
  - **NumPy**: For numerical operations
  - **Scikit-learn**: For model training, preprocessing, and evaluation
  - **Joblib**: For saving and loading trained models

---

## 💻 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nikpirat/house-price-predictor.git
cd house-price-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place your CSV files in the `data/raw/` folder:
- `houses.csv` – training data with `SalePrice` column
- `new_houses.csv` – new data to predict prices (no `SalePrice` needed)

### 4. Run the Pipeline

```bash
python main.py
```

---

## 📊 Model Evaluation Metrics

After training, the model achieved the following performance on the test set:

🔸 **MSE  (Mean Squared Error)**: 98,023,424.88  
🔸 **MAE  (Mean Absolute Error)**: 7,962.18  
🔸 **R² Score**: 0.8206  

These indicate that the model explains approximately 82% of the variance in house prices and, on average, makes predictions within ~$8,000 of the actual price.

---

## 📬 Output Example

**Sample Predictions vs Actual:**

```
🔹 Predicted: 189,559.61  |  Actual: 202,403.00
🔹 Predicted: 186,394.71  |  Actual: 191,789.00
🔹 Predicted: 196,208.37  |  Actual: 196,149.00
🔹 Predicted: 201,218.15  |  Actual: 214,476.00
🔹 Predicted: 199,158.18  |  Actual: 192,714.00
```

**New Predictions from CSV:**

```
OverallQual  GrLivArea  YearBuilt  Neighborhood  PredictedPrice
---------------------------------------------------------------
5            984        2011       OldTown       135,256.60
7           1799        2008       OldTown       192,680.88
5           1359        2004       Somerst       148,748.13
```
