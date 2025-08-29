
#Advanced Sentiment Analysis Project

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')


#Load and Explore Dataset


df = pd.read_csv("Tweets.csv")
print("Dataset Information:")
print(f"Total number of tweets: {len(df)}")
print(f"Class distribution:\n{df['airline_sentiment'].value_counts()}")
print(f"Missing values: {df.isnull().sum().sum()}")

df = df[["text", "airline_sentiment"]].copy()
df.dropna(inplace=True)


#Advanced Data Preprocessing


stop_words = set(ENGLISH_STOP_WORDS)

def advanced_clean_text(text):
    """Advanced text cleaning function"""
    if pd.isna(text):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'@\w+', '', text)
    
    text = re.sub(r'<.*?>', '', text)
    
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 2]
    
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(advanced_clean_text)

df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

print(f"\nNumber of tweets after cleaning: {len(df)}")


#Data Visualization


plt.figure(figsize=(15, 10))

#Class distribution

plt.subplot(2, 3, 1)
df['airline_sentiment'].value_counts().plot(kind='bar', color=['red', 'gray', 'green'])
plt.title('Sentiment Distribution')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)

#Tweet length distribution

plt.subplot(2, 3, 2)
df['text_length'] = df['clean_text'].str.len()
plt.hist(df['text_length'], bins=50, alpha=0.7)
plt.title('Tweet Length Distribution')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')

#Average tweet length per class

plt.subplot(2, 3, 3)
avg_lengths = df.groupby('airline_sentiment')['text_length'].mean()
avg_lengths.plot(kind='bar', color=['red', 'gray', 'green'])
plt.title('Average Tweet Length')
plt.ylabel('Number of Characters')
plt.xticks(rotation=45)

#Most frequent words for each class

sentiments = ['negative', 'neutral', 'positive']
colors = ['red', 'gray', 'green']

for i, (sentiment, color) in enumerate(zip(sentiments, colors)):
    plt.subplot(2, 3, 4+i)
    text_data = " ".join(df[df['airline_sentiment'] == sentiment]['clean_text'])
    if len(text_data) > 0:
        words = text_data.split()
        word_freq = pd.Series(words).value_counts().head(10)
        word_freq.plot(kind='barh', color=color, alpha=0.7)
        plt.title(f'{sentiment.capitalize()} - Most Frequent Words')
        plt.xlabel('Frequency')
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


#Train and Test Sets


X = df["clean_text"]
y = df["airline_sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


#Advanced TF-IDF Vectorization


vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Use unigrams and bigrams
    min_df=2,            # Must appear in at least 2 documents
    max_df=0.95          # Must not appear in more than 95% of documents
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Number of TF-IDF features: {X_train_tfidf.shape[1]}")


#Advanced Models


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}


#Model Training, Cross-Validation and Evaluation


print("\n" + "="*80)
print("MODEL RESULTS")
print("="*80)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
    
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    results[name] = {
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": classification_report(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred)
    }
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


#Comprehensive Visualization


plt.figure(figsize=(20, 12))

plt.subplot(2, 4, 1)
accuracies = [results[m]["accuracy"] for m in results]
bars = plt.bar(list(results.keys()), accuracies, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.subplot(2, 4, 2)
cv_means = [results[m]["cv_mean"] for m in results]
cv_stds = [results[m]["cv_std"] for m in results]
plt.bar(list(results.keys()), cv_means, yerr=cv_stds, 
        color=['skyblue', 'lightgreen', 'salmon', 'gold'], alpha=0.7)
plt.title('Cross-Validation Accuracy')
plt.ylabel('CV Accuracy')
plt.xticks(rotation=45)

plt.subplot(2, 4, 3)
f1_scores = [results[m]["f1"] for m in results]
plt.bar(list(results.keys()), f1_scores, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
plt.title('F1-Score Comparison')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)

plt.subplot(2, 4, 4)
precisions = [results[m]["precision"] for m in results]
recalls = [results[m]["recall"] for m in results]
plt.scatter(precisions, recalls, s=100, c=['skyblue', 'lightgreen', 'salmon', 'gold'])
for i, name in enumerate(results.keys()):
    plt.annotate(name, (precisions[i], recalls[i]), xytext=(5, 5), textcoords='offset points')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision vs Recall')

for i, (name, result) in enumerate(results.items()):
    plt.subplot(2, 4, 5+i)
    sns.heatmap(result["cm"], annot=True, fmt='d', cmap='Blues',
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title(f'{name}\nConfusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

plt.tight_layout()
plt.show()


#Select Best Model and Detailed Analysis


best_model_name = max(results.keys(), key=lambda x: results[x]["f1"])
best_model = models[best_model_name]

print(f"\nBEST MODEL: {best_model_name}")
print("="*50)
print(f"Test Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"CV Accuracy: {results[best_model_name]['cv_mean']:.4f}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
print("\nDetailed Classification Report:")
print(results[best_model_name]["report"])


#Feature Importance 


if hasattr(best_model, 'coef_'):
    feature_names = vectorizer.get_feature_names_out()
    
    if len(best_model.coef_.shape) > 1 and best_model.coef_.shape[0] > 2:
        coef = best_model.coef_[0] 
    else:
        coef = best_model.coef_[0] if len(best_model.coef_.shape) > 1 else best_model.coef_
    
    top_pos_indices = coef.argsort()[-20:][::-1]
    top_neg_indices = coef.argsort()[:20]
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    pos_features = [feature_names[i] for i in top_pos_indices]
    pos_scores = [coef[i] for i in top_pos_indices]
    plt.barh(pos_features, pos_scores, color='green', alpha=0.7)
    plt.title('Most Important Positive Features')
    plt.xlabel('Coefficient')
    
    plt.subplot(1, 2, 2)
    neg_features = [feature_names[i] for i in top_neg_indices]
    neg_scores = [coef[i] for i in top_neg_indices]
    plt.barh(neg_features, neg_scores, color='red', alpha=0.7)
    plt.title('Most Important Negative Features')
    plt.xlabel('Coefficient')
    
    plt.tight_layout()
    plt.show()


#Sample Predictions


sample_texts = [
    "I love this airline! Great service and comfortable seats.",
    "Terrible experience. Flight was delayed for hours.",
    "The flight was okay, nothing special but not bad either."
]

print("\nSAMPLE PREDICTIONS")
print("="*50)

for text in sample_texts:
    clean_sample = advanced_clean_text(text)
    sample_tfidf = vectorizer.transform([clean_sample])
    prediction = best_model.predict(sample_tfidf)[0]
    
    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(sample_tfidf)[0]
        max_proba = max(proba)
        print(f"Text: '{text}'")
        print(f"Prediction: {prediction} (Confidence: {max_proba:.3f})")
    else:
        print(f"Text: '{text}'")
        print(f"Prediction: {prediction}")
    print("-" * 50)

print("\nAnalysis completed!")