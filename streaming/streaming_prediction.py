import joblib
import pandas as pd
import time

model = joblib.load("../kidney_disease_model.pkl")

df = pd.read_csv("../kidney_disease.csv")

X = df.drop(
    columns=["classification", "id"]
)

print("Starting streaming prediction...\n")

for i in range(10):

    record = X.iloc[[i]]

    prediction = model.predict(record)

    print(
        f"Incoming Record {i + 1} "
        f"Prediction: {prediction[0]}"
    )

    time.sleep(1)

print("\nStreaming simulation completed.")
