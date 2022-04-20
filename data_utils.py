import config
import random
from torchtext import datasets
from torchtext import data
import torch


def get_data(tokenizer, seed):

    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token

    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

    max_input_length = tokenizer.max_model_input_sizes[config.base_model]

    def tokenize_and_cut(sentence):
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_input_length - 2]
        return tokens

    text = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)

    label = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(text, label)

    train_data, valid_data = train_data.split(random_state = random.seed(seed))

    return train_data, test_data, valid_data


