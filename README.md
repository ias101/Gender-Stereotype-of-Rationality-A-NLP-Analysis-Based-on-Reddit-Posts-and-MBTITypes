# Gender Stereotype of Rationality: NLP Analysis Based on Reddit Posts and MBTI Types

## 📋 Overview

This project investigates the common gender stereotype that "men are more rational than women" using Natural Language Processing (NLP) techniques. By analyzing Reddit posts and leveraging MBTI personality types (specifically the Thinking vs. Feeling dichotomy), we examine whether textual data supports or contradicts this stereotype.

## ✨ Key Features

- **Text Classification**: Classifies Reddit posts as "Thinking" or "Feeling" based on MBTI personality indicators
- **Multiple Model Comparison**: Evaluates Logistic Regression, Random Forest, SGDClassifier, and LSTM models
- **Gender Analysis**: Predicts Thinking/Feeling classes across male and female authors
- **Advanced NLP Pipeline**: Includes text preprocessing, TF-IDF vectorization, and semantic analysis
- **Statistical Validation**: Compares model performance against baseline accuracy

## 📊 Research Question

**Are men more rational than women?**  
We investigate this by:
1. Training models to detect "Thinking" vs. "Feeling" from text
2. Applying the best-performing model to gender-labeled Reddit data
3. Analyzing the distribution of Thinking/Feeling predictions across genders

## 📁 Project Structure

```
├── data/
│   ├── feeling_thinking_dataset.csv    # Reddit posts labeled as Thinking/Feeling
│   └── gender_dataset.csv              # Reddit posts with gender labels
├── src/
│   ├── data_preprocessing.py           # Text cleaning and preprocessing
│   ├── feature_extraction.py           # TF-IDF vectorization
│   ├── model_training.py               # Model training and evaluation
│   ├── lstm_model.py                   # LSTM implementation with GloVe embeddings
│   └── gender_analysis.py              # Gender prediction analysis
├── results/
│   ├── model_performance.csv           # Performance metrics
│   ├── gender_predictions.png          # Visualization of results
│   └── word_clouds/                    # Visual representations of classes
├── notebooks/
│   └── exploratory_analysis.ipynb      # Jupyter notebook for data exploration
└── requirements.txt                    # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gender-stereotype-rationality.git
cd gender-stereotype-rationality
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download additional resources:
```bash
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
```

## 📈 Methodology

### 1. Data Preprocessing
- **Noise Reduction**: Removal of emojis, HTML tags, punctuation, stopwords, and special characters
- **Spelling Correction**: Using Spello library
- **Lemmatization**: Using spaCy for word normalization
- **Text Cleaning**: Removal of duplicate and repeated patterns

### 2. Feature Extraction
- **TF-IDF Vectorization**: With parameter tuning (lowercase=True, max_features=1000)
- **Word Embeddings**: GloVe embeddings for LSTM model

### 3. Model Training
Four models were trained and evaluated:
- **Logistic Regression**
- **Random Forest Classifier**
- **SGDClassifier** (selected as best-performing)
- **LSTM** with pre-trained GloVe embeddings

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Comparison against baseline (66% for Thinking, 34% for Feeling)

## 📊 Results

### Model Performance
- **Best Model**: SGDClassifier with 82% accuracy
- **Baseline Comparison**: Significant improvement over baseline for both classes
- **Parameter Settings**: lowercase=True, max_features=1000

### Gender Analysis Findings
- **Males**: 91.07% predicted as "Thinking", 8.93% as "Feeling"
- **Females**: 72.29% predicted as "Thinking", 27.71% as "Feeling"

**Key Insight**: While both genders lean toward "Thinking," females show a more balanced distribution, contradicting the stereotype that men are more rational.

## 🚀 How to Run

### 1. Data Preparation
```bash
python src/data_preprocessing.py
```

### 2. Model Training
```bash
python src/model_training.py
```

### 3. Gender Analysis
```bash
python src/gender_analysis.py
```

### 4. Jupyter Notebook
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## 📝 Key Findings

1. **Model Performance**: SGDClassifier outperformed other models with 82% accuracy
2. **Gender Patterns**: Both genders show "Thinking" preference, but females are more balanced
3. **Stereotype Contradiction**: Findings contradict the stereotype that men are more rational than women
4. **Textual Indicators**: Words like "think," "say," and "one" appear frequently in both classes

## ⚠️ Limitations & Future Work

### Current Limitations
- Dataset imbalance (1366 Thinking vs. 701 Feeling posts)
- Limited parameter exploration for models
- Potential social desirability bias in self-reported data

### Future Improvements
- Expand dataset size and diversity
- Experiment with more advanced models (Transformers, BERT)
- Include more nuanced gender categories
- Investigate cross-cultural variations
- Explore temporal trends in gender stereotypes

## 📚 References

Key references include:
- Briggs-Myers & Myers (1995) on MBTI personality types
- Plank & Hovy (2015) on personality detection from Twitter
- Hoyle et al. (2019) on gendered language in literature
- Sladek et al. (2010) on age and gender differences in thinking styles

## 👥 Authors

**Iris Shi** (1778676) & **Jikun Shen** (1833847)  
*Academic project on Gender Stereotypes and NLP*

## 📄 License

This project is for academic research purposes. Please cite appropriately if using the methodology or findings.
