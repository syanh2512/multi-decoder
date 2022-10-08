import torch
import torch.nn as nn
import joblib
from data_processor import Preprocessor
from model import EncoderGRU



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class SampleGenerator():
    def __init__(self, train_data, hidden_vector_dict, word2idx):
        self.data = train_data
        self.processor = Preprocessor(hidden_vector_dict, word2idx)

    def load_dataset(self):
        self.hidden_vector_dict = joblib.load('jissen/results/CIDDS-001/hidden-vector.joblib') 
        self.word2idx = joblib.load('jissen/data/processed/word2idx')
        self.dataset = CIDDS_001_Loader() #editing


    def generate_sample(self, ) #editing


hidden_vector_dict = joblib.load("jissen/results/CIDDS-001/hidden-vector.joblib") # hidden_vector_dict = joblib.load("1-1-32/20220710010024/hidden-vector.joblib")
for i,_ in enumerate(hidden_vector_dict):
    hidden_vector_dict[i].requires_grad = False
print(f'{hidden_vector_dict[79499]} \n size of hidden vector dict: {len(hidden_vector_dict)}')


batch_size = 100
n_iters = 4000
n_epochs = n_iters / (len(hidden_vector_dict) / batch_size)
n_epochs = int(n_epochs)

input_dim = 20
hidden_dim = 128
n_layers = 2
output_dim = 20

model = EncoderGRU(input_dim, hidden_dim, n_layers, output_dim)

learning_rate = 0.05

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
