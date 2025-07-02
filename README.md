## NLP-using-TensorFlow-on-Amazon-Reviews
## Recent changes: 
* Integrated LLaMA 3.2 (via Ollama) for improved review summarization and aspect-based sentiment analysis, replacing the previous T5-base summarizer, spaCy + lexicon-based aspect extractor, and DistilBERT-based aspect clause-level sentiment classifier.
* prevously->absa.py and summary.py
* now->absa_with_llm.py and summary_with_llm.py
## Introduction

Sentiment Analysis of Amazon Reviews is an NLP task that involves analyzing customer feedback to determine sentiment polarity (positive or negative). This project aims to extract insights from Amazon reviews using deep learning and NLP techniques. Sentiment analysis is widely used in e-commerce, marketing, and customer feedback analysis to enhance business decision-making.

## Abstract

This project focuses on sentiment analysis of Amazon reviews using TensorFlow and NLP techniques. It preprocesses textual data, cleans it, and applies machine learning models to classify reviews as either positive or negative. The goal is to create an efficient and accurate sentiment analysis system using deep learning. The following sections outline the key aspects of the project, including data preprocessing, model training, evaluation, and application.

## Technology

- **Python**: The primary programming language used for data processing and model training.
- **Pandas & NumPy**: Used for data manipulation and preprocessing.
- **NLTK**: Utilized for text processing, including stopword removal and stemming.
- **Scikit-learn**: Provides data preprocessing utilities and model evaluation metrics.
- **TensorFlow**: Used for building and training deep learning models for sentiment classification.
- **Google Colab**: Provides a cloud-based environment for executing the code.

## Uses and Applications

Sentiment analysis of Amazon reviews has various practical applications, including:

- **Customer Feedback Analysis**: Understanding customer sentiment to improve products and services.
- **E-commerce**: Enhancing product recommendations and understanding customer preferences.
- **Marketing Insights**: Identifying trends and customer satisfaction levels.
- **Automated Moderation**: Filtering spam and inappropriate reviews from online platforms.

## Steps to Build

### 1. Data Collection
   - Load the dataset containing Amazon product reviews.
   - Dataset: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=train.csv
   - Clean and preprocess the dataset by handling missing values.

### 2. Data Preprocessing
   - Remove unnecessary columns (e.g., reviewer ID, timestamps).
   - Merge text fields for better representation.
   - Apply text preprocessing techniques (stopword removal, stemming, tokenization).
   - Convert overall ratings into binary sentiment labels (positive or negative).

### 3. Model Training
   - Split the dataset into training and testing sets.
   - Convert text into numerical representations using word embeddings.
   - Train a deep learning model (e.g., LSTM, CNN) using TensorFlow.

### 4. Model Evaluation
   - Evaluate model performance using accuracy, precision, recall, and F1-score.
   - Optimize hyperparameters to improve accuracy.

### 5. Testing and Deployment
   - Test the trained model on new reviews.
   - Deploy the model using a web application or API for real-time sentiment analysis.

## Conclusion

This project demonstrates how sentiment analysis can be applied to Amazon reviews using deep learning and NLP techniques. By leveraging TensorFlow and NLP tools, we can efficiently classify customer reviews, providing valuable insights for businesses and consumers. The approach can be extended to other e-commerce platforms, enhancing customer experience and decision-making.

## Summary

## 1. **Input**: The dataset consists of Amazon product reviews containing text descriptions and overall ratings.
## 2. **Processing**: The reviews are cleaned, preprocessed, and transformed into numerical features. A deep learning model is trained to classify sentiment.
## 3. **Output**: The model predicts whether a review is positive or negative and provides sentiment insights for further analysis.
![Screenshot 2025-07-02 124752](https://github.com/user-attachments/assets/99650dc9-68c7-403a-b058-9b4c1a7ad84f)


## contact me
  * email -> vivekpippalla@gmail.com  

