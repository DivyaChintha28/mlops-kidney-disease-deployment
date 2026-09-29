import joblib
import pandas as pd

model = joblib.load("../kidney_disease_model.pkl")

df = pd.read_csv("../kidney_disease.csv")

print("Number of records:", len(df))

ids = df["id"]

X = df.drop(
    columns=["classification", "id"]
)

predictions = model.predict(X)

output = pd.DataFrame({
    "id": ids,
    "prediction": predictions
})

output.to_csv(
    "batch_predictions.csv",
    index=False
)

print("\nBatch prediction completed.")

print("\nFirst 10 predictions:")
print(output.head(10))

print("\nOutput file created: batch_predictions.csv")
