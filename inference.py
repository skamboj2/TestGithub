import pickle
import numpy as np

# Load the model
with open('dt_classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example: Make inference on a single sample
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Replace with your actual features
prediction = model.predict(sample)
prediction_proba = model.predict_proba(sample)

print(f"Prediction: {prediction}")
print(f"Prediction Probability: {prediction_proba}")

# Example: Make inference on multiple samples
samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4]
])
predictions = model.predict(samples)
print(f"Predictions: {predictions}")