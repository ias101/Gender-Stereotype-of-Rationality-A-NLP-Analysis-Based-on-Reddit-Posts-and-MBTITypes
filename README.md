# Gender Stereotype of Rationality: NLP Analysis Based on Reddit Posts and MBTI Types

## Project Overview
This study investigates a common gender stereotype: "men are more rational than women" using Natural Language Processing (NLP) techniques. Based on Reddit posts and the Myers-Briggs Type Indicator (MBTI) "Thinking" vs. "Feeling" dichotomy, we train text classification models to predict rationality-associated text categories and analyze their distribution across genders.

## Key Findings
- **SGDClassifier** performed best in classifying "Thinking" vs. "Feeling" posts (accuracy: 0.84).
- Prediction results show: **91.07% of male-authored posts were classified as "Thinking"**, while **72.29% of female-authored posts were classified as "Thinking"**.
- Females show a more balanced distribution between "Thinking" and "Feeling", though still leaning toward "Thinking".
- **Results contradict the common gender rationality stereotype**, not supporting "men are more rational than women".

## Datasets
Two Reddit datasets were used:
1. **Feeling and Thinking Dataset**: For training/testing "Thinking" vs. "Feeling" classification models.
   - 2,067 posts (1,366 "Thinking", 701 "Feeling").
2. **Gender Dataset**: For gender-based rationality analysis.
   - 2,401 posts (1,281 female, 1,120 male).

## Data Preprocessing
- Noise removal: numbers, emojis, HTML tags, stopwords, repeated words, etc.
- Spelling correction using Spello.
- Lemmatization using spaCy.

## Models & Experimentation
Four models were tested and compared:
1. Logistic Regression
2. Random Forest Classifier
3. SGDClassifier
4. LSTM (with pre-trained GloVe embeddings)

Hyperparameter tuning via GridSearchCV yielded optimal TF-IDF parameters: `lowercase=True`, `max_features=1000`.

## Results & Evaluation
| Model | Accuracy | Note |
|-------|----------|------|
| SGDClassifier | 0.82 | Best performer |
| Logistic Regression | 0.81 | Good performance |
| Random Forest | 0.75 | Low recall for "Feeling" class |
| LSTM | 0.60 | Poor performance |

# [Paper](./final_report.pdf)
