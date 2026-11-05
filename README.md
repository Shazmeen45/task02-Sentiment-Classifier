# Multi-Class Sentiment Classifier

## Progree AI Internship – Task 2

A machine learning project that classifies text into three sentiment categories: **Positive, Negative, and Neutral**. This project uses the Twitter US Airline Sentiment dataset and applies Natural Language Processing (NLP) techniques with TF-IDF and Logistic Regression.

## Project Overview

The purpose of this project is to build an intelligent sentiment analysis system that can understand text and predict its sentiment.

The classifier can be used to analyze customer feedback, reviews, and social media messages.

## Dataset

**Dataset:** Twitter US Airline Sentiment

The dataset contains tweets about US airlines labeled with one of three sentiments:

- Negative
- Neutral
- Positive

### Dataset Distribution

| Sentiment | Number of Tweets |
| --------- | ---------------: |
| Negative  |            9,178 |
| Neutral   |            3,099 |
| Positive  |            2,363 |
| **Total** |       **14,640** |

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- TF-IDF Vectorization
- Logistic Regression

## Project Workflow

1. Load the Twitter US Airline Sentiment dataset.
2. Clean and preprocess text data.
3. Convert text into lowercase.
4. Remove unnecessary characters and stopwords.
5. Apply lemmatization.
6. Split data into training and testing sets.
7. Convert text into numerical features using TF-IDF.
8. Train a Logistic Regression model.
9. Evaluate model performance.
10. Test the model on custom text.
11. Save the trained model and TF-IDF vectorizer.

## Model Details

**Algorithm:** Logistic Regression

**Feature Extraction:** TF-IDF Vectorizer

**Train-Test Split:** 80% training, 20% testing

**Random State:** 42

## Model Performance

| Metric         |  Score |
| -------------- | -----: |
| Accuracy       | 76.20% |
| Macro F1-Score | 71.53% |

The model was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

## Project Files

- `Task2_Sentiment_Classifier.ipynb` — Complete Google Colab notebook
- `sentiment_model.pkl` — Trained Logistic Regression model
- `tfidf_vectorizer.pkl` — Saved TF-IDF vectorizer
- `Task2_Sentiment_Classifier_Report.pdf` — Project report

## How to Use

### 1. Install Required Libraries

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn joblib
```

### 2. Load the Saved Model

```python
import joblib

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
```

### 3. Predict Sentiment

Use the same preprocessing function from the notebook before making predictions.

```python
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    text_vector = tfidf.transform([cleaned_text])
    prediction = model.predict(text_vector)
    return prediction[0]

print(predict_sentiment("I love the airline service!"))
```

## Future Improvements

- Experiment with advanced machine learning algorithms.
- Improve neutral sentiment classification.
- Explore deep learning and transformer-based models.
- Develop a web-based sentiment analysis application.
- Improve preprocessing and model performance.

## Author

**Shazmeen**

## Internship

**Progree AI Internship Program — Task 2**

