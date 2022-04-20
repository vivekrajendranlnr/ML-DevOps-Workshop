import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModel
from SentimentAnalyser import SentimentAnalyzer
import config

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
    tokens = tokenizer.tokenize(sentence)
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    tokens = tokens[:config.max_input_length - 2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


sentiment_analyser = FastAPI()

origins = ["*"]

sentiment_analyser.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@sentiment_analyser.post("/")
async def get_sentiments(text: str):
    prediction = predict_sentiment(text)
    if prediction >= 0.5:
        response = {"result": "positive"}
    else:
        response = {"result": "negative"}
    return response


if __name__ == "__main__":
    uvicorn.run("service:sentiment_analyser", host="0.0.0.0", port=9001)

