import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.config import MODEL_PATH

def evaluate_model(X_test, y_test, print_samples=True):
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n📊 Model Evaluation on Test Set:")
    print(f"🔸 MSE  (Mean Squared Error): {mse:.2f}") # Shows average of the squared differences between predicted and actual prices.
    # Error size, penalizes big ones
    # Unit: Squared target units
    # Smaller is better

    print(f"🔸 MAE  (Mean Absolute Error): {mae:.2f}") # Shows average of the absolute difference between predicted and actual prices.
    # Average error size
    # Unit: Same as target (e.g., $)
    # Smaller is better

    print(f"🔸 R² Score: {r2:.4f}") # Coefficient of Determination - Measures how well your model explains the variance in the target variable.
    # R² = 0.8206 → your model explains 82% of the variability in house prices.
    # R² = 1 → perfect model
    # R² = 0 → no better than predicting the mean
    # R² < 0 → worse than mean prediction
    # Overall effectiveness score

    if print_samples:
        print("\n🧪 Sample Predictions vs Actual:")
        for i in range(5):
            print(f"  🔹 Predicted: {y_pred[i]:,.2f}  |  Actual: {y_test.iloc[i]:,.2f}")
