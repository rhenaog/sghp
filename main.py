import argparse
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import secrets
from dataset import get_dataloader
from gated_tpp import gated_tpp,count_parameters

DATASET_PATHS = {'sin_hawkes': '../data/simulated/sin_hawkes/', 'power_hawkes': '../data/simulated/power_hawkes/',
                'poisson': '../data/simulated/poisson/','exp_hawkes':'../data/simulated/exp_hawkes/','sin_hawkes_2':'../data/simulated/sin_hawkes_2/',
                 '2_d_hawkes': '../data/simulated/2_d_hawkes/', 'mimic': '../data/mimic/',
                 'stackOverflow': '../data/stackoverflow/', 'retweet': '../data/retweet/'}

DATASET_EVENT_TYPES = {'sin_hawkes': 1, 'power_hawkes': 1,'poisson':1, '2_d_hawkes': 2, 'mimic': 75, 'stackOverflow': 22,
                       'retweet': 3,'exp_hawkes':1,'sin_hawkes_2':1}



parser = argparse.ArgumentParser()

parser.add_argument('-data', type=str, default='2_d_hawkes')
parser.add_argument('-model', type=int, default=1)
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=20)
parser.add_argument('-d_model', type=int, default=32)
parser.add_argument('-p_norm', type=float, default=1)
parser.add_argument('-sigma', type=float, default=1)

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-l2', type=float, default=0.0000)
parser.add_argument('-save', type=bool, default=True)
parser.add_argument('-normalize', type=int, default=0)

params = parser.parse_args()
params.normalize = True if params.normalize == 1 else False

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

params.device = torch.device(device)
print('[Info] parameters: {}'.format(params))

data_path = DATASET_PATHS[params.data]
num_types = DATASET_EVENT_TYPES[params.data]

with open(data_path + 'train.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin-1')
    train_data = data['train']

with open(data_path + 'dev.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin-1')
    dev_data = data['dev']

with open(data_path + 'test.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin-1')
    test_data = data['test']

t_max = max([seq[-1]['time_since_start'] for data in [train_data, dev_data, test_data] for seq in data])
if not params.normalize:
    t_max = 1

trainloader = get_dataloader(train_data, params.batch_size, shuffle=True,t_max = t_max)
testloader = get_dataloader(test_data, 1, shuffle=False,t_max = t_max)  # 1 makes it easy to calculate RMSE
valloader = get_dataloader(dev_data, 1, shuffle=False,t_max = t_max)
valid_events = 0
test_events = 0
train_events = 0
for seq in valloader.dataset.event_type:
    valid_events += len(seq)
for seq in trainloader.dataset.event_type:
    train_events += len(seq)
for seq in testloader.dataset.event_type:
    test_events  += len(seq)

model = gated_tpp(num_types, params.d_model,dropout=params.dropout)

optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                       params.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=params.l2)

model = model.to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
print(count_parameters(model))


train_losses = []
validation_losses = []
parameters = []

print("the number of trainable parameters: " + str(count_parameters(model)))
start_time = time.time()
for epoch in range(params.epoch):
    train_epoch_loss, _ = model.train_epoch(trainloader, optimizer, params)

    train_loss = train_epoch_loss / train_events
    total_time = time.time() - start_time
    average_time = total_time/(epoch+1)

    print(f'Epoch:{epoch}, Train Loss:{train_loss:.6f}, Valid Loss:{train_loss:.6f}, Test Loss:{train_loss:.6f},Time per Epoch :{average_time:.6f}')

test_epoch_loss, _, test_f1_score, test_last_RMSE, test_accuracy = model.validate_epoch(testloader,
                                                                                        device=params.device)
valid_epoch_loss, _, val_f1_score, val_last_rmse, val_accuracy = model.validate_epoch(valloader,
                                                                                      device=params.device )

print(f' Test Last Event RMSE:{test_last_RMSE:.4f}, Test Last Event F-1 Score:{test_f1_score:.4f},')


model_name =secrets.token_hex(5)
if params.save:
    torch.save(model.state_dict(), 'trained_models/' + model_name + '.pt')
print(f'Saved Model Name:{model_name}')
