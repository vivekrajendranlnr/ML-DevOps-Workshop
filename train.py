import config
import torch
from transformers import AutoTokenizer, AutoModel
from SentimentAnalyser import SentimentAnalyzer
from data_utils import get_data
from train_utils import train, evaluate, epoch_time
from torchtext import data
import torch.optim as optim
import random
import numpy as np
import time

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(config.base_model)
b_model = AutoModel.from_pretrained(config.base_model)

train_data, test_data, valid_data = get_data(tokenizer, seed=config.seed)

model = SentimentAnalyzer(config.base_model,
                          config.hidden_dim,
                          config.output_dim,
                          config.n_layers,
                          config.bidirectional,
                          config.dropout)

for name, param in model.named_parameters():
    if name.startswith('b_model'):
        param.requires_grad = False

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=config.batch_size,
    device=device)


optimizer = optim.Adam(model.parameters())
criterion = torch.nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

best_valid_loss = float('inf')

for epoch in range(config.epochs):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), config.model_path)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

