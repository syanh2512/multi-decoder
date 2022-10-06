import torch
import torch.nn as nn
import joblib



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


hidden_vector_dict = joblib.load("jissen/results/CIDDS-001/hidden-vector.joblib")
print(len(hidden_vector_dict))

