# NLP-using-TensorFlow-on-Amazon-Reviews
# **NLP Using TensorFlow on Amazon Reviews**

![Deep Learning](https://img.shields.io/badge/Deep_Learning-TensorFlow-orange)
![NLP](https://img.shields.io/badge/NLP-Text_Analysis-blue)

## **📌 Project Overview**
This project analyzes Amazon product reviews using **Natural Language Processing (NLP) and Deep Learning (TensorFlow)**. The system provides three key outputs:

✅ **Sentiment Analysis** → Classifies reviews as **Positive** or **Negative**.  
✅ **Aspect-Based Sentiment Analysis** → Extracts key aspects and determines their sentiment.  
✅ **Fake Review Detection** → Identifies if a review is **Fake, Suspicious, or Genuine**.  

By automating review analysis, this project enhances the credibility of online reviews and provides structured feedback insights.

---

## **📂 Dataset & Preprocessing**
- The dataset used is the **best available Amazon review dataset** with labeled sentiment and fake review detection data.
- **Preprocessing Steps:**
  - Convert text to lowercase
  - Remove special characters and numbers
  - Tokenization
  - Stopword removal
  - Lemmatization

The **preprocessed dataset is saved as `cleaned_amazon_reviews.csv`**.

