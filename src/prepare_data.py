from datasets import load_dataset


# Завантажуємо dataset
dataset = load_dataset(
    "pszemraj/goodreads-bookgenres"
)

train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]


print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# Перевіряємо перший приклад
print("\nFirst book:")
print("Title:", train[0]["Book"])
print("Description:", train[0]["Description"])
print("Genres:", train[0]["Genres"])


# Перевіряємо пропуски
def check_missing(data, name):
    missing_description = sum(
        not description or not description.strip()
        for description in data["Description"]
    )

    print(f"\n{name}:")
    print("Missing descriptions:", missing_description)


check_missing(train, "Train")
check_missing(validation, "Validation")
check_missing(test, "Test")


# Кількість жанрів
num_genres = len(train[0]["Genres"])

print("\nNumber of genres:", num_genres)