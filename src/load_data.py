from datasets import load_dataset


dataset = load_dataset(
    "pszemraj/goodreads-bookgenres"
)

print(dataset)

print("\nTrain:")
print(dataset["train"])

print("\nValidation:")
print(dataset["validation"])

print("\nTest:")
print(dataset["test"])

print("\nColumns:")
print(dataset["train"].column_names)

print("\nFirst book:")
print(dataset["train"][0])