import joblib
import torch

with torch.no_grad():
    hidden_vector_dict = joblib.load("jissen/results/CIDDS-001/hidden-vector.joblib")
    print(hidden_vector_dict)