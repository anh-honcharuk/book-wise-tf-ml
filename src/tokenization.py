from datasets import load_dataset
from transformers import BertTokenizer


# Завантажуємо dataset
dataset = load_dataset(
    "pszemraj/goodreads-bookgenres"
)

train = dataset["train"]


# Завантажуємо BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


# Беремо опис першої книги
description = train[0]["Description"]

print("Original text:")
print(description)


# Токенізація
tokens = tokenizer(
    description,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="tf"
)


print("\nTokenized:")
print(tokens)


print("\nInput IDs:")
print(tokens["input_ids"])


print("\nAttention mask:")
print(tokens["attention_mask"])