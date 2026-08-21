# 📚 BookWise — Розумна система рекомендації книг

BookWise — ML-система для аналізу книг, яка використовує методи **NLP, Computer Vision та Recommendation Systems**.

Система аналізує текстовий опис книги, її метадані та обкладинку для:

* класифікації книг за жанрами;
* генерації текстових описів;
* пошуку схожих книг;
* аналізу стилю обкладинки;
* надання рекомендацій через веб-інтерфейс.

## Архітектура

Проєкт складається з декількох незалежних модулів:

### 1. Збір та підготовка даних

Для отримання та обробки інформації про книги використовуються:

* `Requests` — HTTP-запити;
* `BeautifulSoup` — парсинг HTML;
* `Pandas` — обробка табличних даних;
* готові датасети з інформацією про книги, рейтинги та відгуки.

Дані можуть містити назву, автора, опис, жанр, рейтинг та зображення обкладинки.

### 2. Класифікація жанрів

Для визначення жанру книги за її описом використовується **BERT**.

Pipeline:

```text
Опис книги
    ↓
BertTokenizer
    ↓
Токени
    ↓
BERT
    ↓
Передбачений жанр
```

Використані технології:

* TensorFlow;
* Hugging Face Transformers;
* BERT;
* `SparseCategoricalCrossentropy`;
* Adam optimizer.

Підтримується багатокласова класифікація жанрів.

### 3. Рекомендаційна система

BookWise використовує контентний підхід для пошуку схожих книг.

Текстові характеристики книг перетворюються у TF-IDF-вектори, після чого книги порівнюються за допомогою **cosine similarity**.

```text
Опис книг
    ↓
TF-IDF
    ↓
Вектори
    ↓
Cosine Similarity
    ↓
Найбільш схожі книги
```

Також передбачена можливість використання **collaborative filtering** за допомогою TensorFlow Recommenders.

Таким чином, система може бути розширена до гібридної рекомендаційної системи.

### 4. Генерація описів

Для генерації текстових описів використовується **GPT-2**.

```text
Prompt
  ↓
GPT-2
  ↓
Згенерований текст
```

Параметри генерації можуть включати:

* `temperature`;
* `top_k`;
* `top_p`;
* `max_length`.

### 5. Аналіз обкладинок

Для аналізу зображень використовується **Convolutional Neural Network (CNN)** на базі TensorFlow/Keras.

Модель містить:

* `Conv2D`;
* `MaxPooling2D`;
* `Flatten`;
* `Dense`;
* `Dropout`.

Модель класифікує обкладинки за попередньо визначеними стилями.

```text
Зображення обкладинки
        ↓
      CNN
        ↓
Витягування ознак
        ↓
Класифікація стилю
```

### 6. Веб-інтерфейс

Для демонстрації системи використовується **Streamlit**.

Користувач може:

1. ввести опис книги;
2. запустити аналіз;
3. отримати передбачений жанр;
4. отримати рекомендації схожих книг.

## Технології

| Категорія        | Технології                                         |
| ---------------- | -------------------------------------------------- |
| Language         | Python                                             |
| NLP              | BERT, GPT-2, Transformers                          |
| Machine Learning | TensorFlow, Scikit-learn                           |
| Computer Vision  | CNN, TensorFlow/Keras                              |
| Recommendations  | TF-IDF, Cosine Similarity, TensorFlow Recommenders |
| Data Processing  | Pandas, NumPy                                      |
| Data Collection  | Requests, BeautifulSoup                            |
| Web Interface    | Streamlit                                          |

## Встановлення

Клонуйте репозиторій:

```bash
git clone <repository-url>
cd BookWise
```

Створіть віртуальне середовище:

```bash
python -m venv .venv
```

Активуйте його.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Встановіть залежності:

```bash
pip install -r requirements.txt
```

## Запуск

Для запуску Streamlit-застосунку:

```bash
streamlit run app.py
```

Після запуску відкрийте адресу, яку покаже Streamlit у терміналі.

## Приклад роботи

Користувач вводить опис:

```text
A young wizard discovers a magical world
and learns to control his powers.
```

Система може визначити:

```text
Genre: Fantasy
```

та запропонувати книги зі схожим описом.

## Структура проєкту

```text
BookWise/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── genre_classifier/
│   ├── recommender/
│   ├── description_generator/
│   └── cover_analyzer/
│
├── src/
│   ├── data_collection/
│   ├── genre_classification/
│   ├── recommendation/
│   ├── description_generation/
│   └── cover_analysis/
│
├── app.py
├── requirements.txt
└── README.md
```

## Основні можливості

* 📖 Аналіз текстового опису книги
* 🏷️ Класифікація жанру
* 🔎 Пошук схожих книг
* ✍️ Генерація описів
* 🖼️ Аналіз обкладинок
* 🌐 Веб-інтерфейс на Streamlit

## Подальший розвиток

Можливі покращення системи:

* fine-tuning BERT на спеціалізованому датасеті книг;
* використання сучаснішої генеративної моделі;
* покращення CNN за допомогою transfer learning;
* додавання реальних user-item interactions;
* повноцінний hybrid recommender;
* збереження моделей та векторів у production-ready форматі;
* API для інтеграції з іншими застосунками;
* Docker-контейнеризація та deployment.

## License

This project is intended for educational and demonstration purposes.
