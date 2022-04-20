import config
import torch
from transformers import AutoTokenizer, AutoModel
from SentimentAnalyser import SentimentAnalyzer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(config.base_model)
b_model = AutoModel.from_pretrained(config.base_model)

model = SentimentAnalyzer(config.base_model,
                          config.hidden_dim,
                          config.output_dim,
                          config.n_layers,
                          config.bidirectional,
                          config.dropout)

model.load_state_dict(torch.load(config.model_path))
model.to(device)


def predict_sentiment(sentence):
    model.eval()
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:config.max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


predict_sentiment("This workshop is awesome")

