# NLP_restaurant_review

# Restaurant Reviews Analysis

link do Kaggle : https://www.kaggle.com/code/stearlit/nlp-restaurant-review

This Jupyter Notebook contains an analysis of restaurant reviews, including text vectorization, sentiment analysis, topic modeling, and evaluation of the results.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Text Vectorization](#text-vectorization)
- [Analysis and Evaluation](#analysis-and-evaluation)
- [Named Entity Recognition (NER)](#named-entity-recognition-ner)
- [Sentiment Analysis](#sentiment-analysis)
- [Topic Modeling](#topic-modeling)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

This project aims to analyze restaurant reviews to extract meaningful insights. The analysis includes various natural language processing (NLP) techniques such as text vectorization, sentiment analysis, and topic modeling.

## Requirements

The following Python libraries are required to run the notebook:

- pandas
- numpy
- nltk
- gensim
- scikit-learn
- spacy
- textblob
- transformers

You can install these dependencies using pip:

\`\`\`bash
pip install pandas numpy nltk gensim scikit-learn spacy textblob transformers
\`\`\`

Additionally, you need to download the \`spaCy\` model:

\`\`\`bash
python -m spacy download en_core_web_sm
\`\`\`

## Data Preprocessing

The initial steps involve loading the data and preprocessing the text. This includes:

- Tokenization
- Removing stopwords
- Lemmatization/Stemming

## Text Vectorization

Text data is converted into numerical vectors using techniques such as:

- TF-IDF Vectorization
- Word2Vec
- BERT embeddings

Example code snippet for TF-IDF vectorization:

\`\`\`python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
\`\`\`

## Analysis and Evaluation

The analysis section covers the evaluation of the model's performance using metrics such as:

- Precision
- Recall
- F1-Score
- Confusion Matrix

Example code snippet for evaluation:

\`\`\`python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
\`\`\`

## Named Entity Recognition (NER)

NER is performed to identify and categorize entities in the text using \`spaCy\`.

Example code snippet for NER:

\`\`\`python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents:
    print(ent.text, ent.label_)
\`\`\`

## Sentiment Analysis

Sentiment analysis is conducted to determine the sentiment polarity of the reviews using \`TextBlob\` or \`VADER\`.

Example code snippet for sentiment analysis:

\`\`\`python
from textblob import TextBlob

analysis = TextBlob("I love this restaurant!")
print(analysis.sentiment)
\`\`\`

## Topic Modeling

Topic modeling is done using LDA (Latent Dirichlet Allocation) to identify underlying topics in the reviews.

Example code snippet for topic modeling:

\`\`\`python
from gensim import corpora, models

dictionary = corpora.Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary)

for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")
\`\`\`

## Results

The results section presents the findings from the various analyses, including sentiment distribution, prominent topics, and entity recognition outcomes.

## Conclusion

The conclusion summarizes the insights gained from the analysis and suggests possible future improvements or areas of further research.


