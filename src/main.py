import pandas as pd
from src.train_model import train_model
from src.evaluate import evaluate_model
from src.predict import predict_house_batch


def main():
    print("\n🔧 Training model...")
    X_test, y_test = train_model()

    print("\n🧪 Evaluating model on test data...")
    evaluate_model(X_test, y_test)

    print("\n🔮 Predicting prices from new data CSV...")

    # Load new data from CSV (unlabeled, no SalePrice column)
    input_csv_path = "../data/raw/new_houses.csv"  # You must create this CSV
    new_data = pd.read_csv(input_csv_path)

    predictions = predict_house_batch(new_data)

    # Output results
    new_data["PredictedPrice"] = predictions.round(2)
    print("\n📄 Sample Predictions:")
    print(new_data.head())

    # Optionally save to file
    output_path = "../outputs/reports/predictions.csv"
    new_data.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    main()