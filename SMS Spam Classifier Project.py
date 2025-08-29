# SMS Spam Classifier Project
# This notebook develops a machine learning model to classify SMS messages as spam or ham

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("SMS Spam Classifier Project")
print("=" * 50)

# 1. DATA LOADING AND EXPLORATION
print("\n1. DATA LOADING AND EXPLORATION")
print("-" * 35)

df = pd.read_csv('spam.csv', encoding='latin-1')
print("✓ Dataset loaded successfully")

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print("\nFirst 10 rows:")
print(df.head(10))

if df.shape[1] > 2:
    df = df.iloc[:, :2]  
    
df.columns = ['label', 'message']  

print(f"\nCleaned dataset shape: {df.shape}")
print(f"Cleaned columns: {df.columns.tolist()}")

# 2. DATA ANALYSIS
print("\n2. DATA ANALYSIS")
print("-" * 20)

label_counts = df['label'].value_counts()
print("Label distribution:")
print(label_counts)
print(f"\nSpam ratio: {label_counts['spam'] / len(df) * 100:.2f}%")
print(f"Ham ratio: {label_counts['ham'] / len(df) * 100:.2f}%")

print(f"\nMissing data count:")
print(df.isnull().sum())

df['message_length'] = df['message'].str.len()
print(f"\nMessage length statistics:")
print(df.groupby('label')['message_length'].describe())

# 3. DATA VISUALIZATION
print("\n3. DATA VISUALIZATION")
print("-" * 25)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
df['label'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 3, 2)
df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'orange'])
plt.title('Label Proportions')
plt.ylabel('')

plt.subplot(1, 3, 3)
df.boxplot(column='message_length', by='label', ax=plt.gca())
plt.title('Message Length by Label')
plt.suptitle('')
plt.xlabel('Label')
plt.ylabel('Message Length')

plt.tight_layout()
plt.show()

# 4. TEXT PREPROCESSING
print("\n4. TEXT PREPROCESSING")
print("-" * 25)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocesses the text"""
    if pd.isna(text):
        return ""
    
    text = text.lower()
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = re.sub(r'\d+', '', text)
    
    tokens = word_tokenize(text)
    
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

print("Applying text preprocessing...")
df['processed_message'] = df['message'].apply(preprocess_text)

print("\nExample preprocessing results:")
for i in range(3):
    print(f"\nOriginal: {df.iloc[i]['message']}")
    print(f"Processed: {df.iloc[i]['processed_message']}")

# 5. FEATURE EXTRACTION
print("\n5. FEATURE EXTRACTION")
print("-" * 23)

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['processed_message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF feature count: {X_train_tfidf.shape[1]}")

count_vect = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_count = count_vect.fit_transform(X_train)
X_test_count = count_vect.transform(X_test)

print(f"Count Vectorizer feature count: {X_train_count.shape[1]}")

# 6. MODEL DEVELOPMENT
print("\n6. MODEL DEVELOPMENT")
print("-" * 20)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, kernel='linear')
}

results = {}

print("Model training with TF-IDF:")
print("-" * 25)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
    
    results[f"{name}_TFIDF"] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Precision: {precision:.4f}")
    print(f"✓ Recall: {recall:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")
    print(f"✓ CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print("\n\nBest model test with Count Vectorizer:")
print("-" * 40)

best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
best_model_type = best_model_name.split('_')[0]

if best_model_type == 'Naive':
    best_model_type = 'Naive Bayes'

best_model_count = models[best_model_type]
best_model_count.fit(X_train_count, y_train)
y_pred_count = best_model_count.predict(X_test_count)

accuracy_count = accuracy_score(y_test, y_pred_count)
precision_count = precision_score(y_test, y_pred_count)
recall_count = recall_score(y_test, y_pred_count)
f1_count = f1_score(y_test, y_pred_count)

print(f"{best_model_type} (Count Vectorizer):")
print(f"✓ Accuracy: {accuracy_count:.4f}")
print(f"✓ Precision: {precision_count:.4f}")
print(f"✓ Recall: {recall_count:.4f}")
print(f"✓ F1-Score: {f1_count:.4f}")

# 7. PERFORMANCE EVALUATION
print("\n7. PERFORMANCE EVALUATION")
print("-" * 32)

best_model_info = results[best_model_name]
best_predictions = best_model_info['predictions']

print(f"Best model: {best_model_name}")
print(f"Best F1-Score: {best_model_info['f1_score']:.4f}")

print("\nDetailed Classification Report:")
print("-" * 30)
print(classification_report(y_test, best_predictions, 
                          target_names=['Ham', 'Spam']))

cm = confusion_matrix(y_test, best_predictions)
print(f"\nConfusion Matrix:")
print(cm)

print("\nAll Models Performance Comparison:")
print("-" * 45)
comparison_df = pd.DataFrame({
    'Model': [name.replace('_TFIDF', '') for name in results.keys()],
    'Accuracy': [results[name]['accuracy'] for name in results.keys()],
    'Precision': [results[name]['precision'] for name in results.keys()],
    'Recall': [results[name]['recall'] for name in results.keys()],
    'F1-Score': [results[name]['f1_score'] for name in results.keys()],
    'CV Mean': [results[name]['cv_mean'] for name in results.keys()]
})
print(comparison_df.round(4))

# 8. VISUALIZATION
print("\n8. RESULT VISUALIZATIONS")
print("-" * 30)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title(f'Confusion Matrix\n{best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(2, 3, 2)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(comparison_df))
width = 0.2

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, comparison_df[metric], width, 
            label=metric, alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width*1.5, comparison_df['Model'], rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.subplot(2, 3, 3)
cv_means = [results[name]['cv_mean'] for name in results.keys()]
cv_stds = [results[name]['cv_std'] for name in results.keys()]
model_names = [name.replace('_TFIDF', '') for name in results.keys()]

plt.errorbar(range(len(cv_means)), cv_means, yerr=cv_stds, 
             marker='o', capsize=5, capthick=2)
plt.xticks(range(len(model_names)), model_names, rotation=45)
plt.ylabel('Cross-Validation Score')
plt.title('Cross-Validation Scores')
plt.grid(alpha=0.3)

plt.subplot(2, 3, 4)
test_labels = ['Ham' if x == 0 else 'Spam' for x in y_test]
pd.Series(test_labels).value_counts().plot(kind='bar', color=['lightblue', 'orange'])
plt.title('Test Set Label Distribution')
plt.xticks(rotation=0)

plt.subplot(2, 3, 5)
correct_predictions = (y_test == best_predictions)
correct_counts = pd.Series(['Correct' if x else 'Wrong' for x in correct_predictions]).value_counts()
correct_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Prediction Accuracy Distribution')
plt.ylabel('')

plt.subplot(2, 3, 6)
if 'Naive' in best_model_name:
    feature_names = tfidf.get_feature_names_out()
    nb_model = results[best_model_name]['model']
    
    feature_log_probs = nb_model.feature_log_prob_
    spam_features = feature_log_probs[1] - feature_log_probs[0]
    
    top_spam_indices = spam_features.argsort()[-10:][::-1]
    top_spam_features = [feature_names[i] for i in top_spam_indices]
    top_spam_scores = [spam_features[i] for i in top_spam_indices]
    
    plt.barh(range(len(top_spam_features)), top_spam_scores, color='red', alpha=0.7)
    plt.yticks(range(len(top_spam_features)), top_spam_features)
    plt.xlabel('Spam Score')
    plt.title('Most Important Spam Words')
else:
    spam_lengths = df[df['label'] == 'spam']['message'].str.len()
    ham_lengths = df[df['label'] == 'ham']['message'].str.len()
    
    plt.hist(ham_lengths, bins=50, alpha=0.7, label='Ham', color='green', density=True)
    plt.hist(spam_lengths, bins=50, alpha=0.7, label='Spam', color='red', density=True)
    
    plt.axvline(ham_lengths.mean(), color='green', linestyle='--', 
                label=f'Ham Avg: {ham_lengths.mean():.0f}')
    plt.axvline(spam_lengths.mean(), color='red', linestyle='--', 
                label=f'Spam Avg: {spam_lengths.mean():.0f}')
    
    plt.xlabel('Message Length (characters)')
    plt.ylabel('Density')
    plt.title('Message Length Distribution')
    plt.legend()
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 9. SAMPLE APPLICATION
print("\n9. SAMPLE APPLICATION")
print("-" * 20)

def predict_message(message, model, vectorizer):
    """Makes spam/ham prediction for a new message - works with all model types"""
    processed = preprocess_text(message)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    
    try:
        probability = model.predict_proba(vectorized)[0]
        confidence = max(probability) * 100
    except AttributeError:
        if hasattr(model, 'decision_function'):
            decision_score = model.decision_function(vectorized)[0]
            abs_score = abs(decision_score)
            if abs_score > 2.0:
                confidence = min(99, 85 + (abs_score - 2.0) * 5)
            elif abs_score > 1.0:
                confidence = 70 + (abs_score - 1.0) * 15
            else:
                confidence = 50 + abs_score * 20
        else:
            confidence = 85.0
    
    if isinstance(prediction, (int, np.integer)):
        label = 'Spam' if prediction == 1 else 'Ham'
    else:
        label = 'Spam' if prediction == 'spam' else 'Ham'
    
    return label, confidence

best_model = results[best_model_name]['model']
best_vectorizer = tfidf

# Test messages
test_messages = [
    "Congratulations! You've won $1000! Click here to claim now!",
    "Hey, are you free for dinner tonight?",
    "URGENT: Your account will be suspended unless you verify now!",
    "Thanks for helping me move yesterday",
    "FREE OFFER! Limited time only! Call now!",
    "Can you pick up some milk on your way home?"
]

print("Predictions for new messages:")
print("-" * 30)

spam_count = 0
ham_count = 0

for i, message in enumerate(test_messages, 1):
    prediction, confidence = predict_message(message, best_model, best_vectorizer)
    
    emoji = "🚨" if prediction == 'Spam' else "✅"
    confidence_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
    
    print(f"\n{i}. {emoji} Message: '{message}'")
    print(f"   Prediction: {prediction} ({confidence:.1f}% confidence - {confidence_level})")
    
    if prediction == 'Spam':
        spam_count += 1
    else:
        ham_count += 1

print(f"\n" + "="*50)
print("PREDICTION SUMMARY")
print("="*50)
print(f"Total Messages Analyzed: {len(test_messages)}")
print(f"Spam Detected: {spam_count} messages ({spam_count/len(test_messages)*100:.1f}%)")
print(f"Ham Messages: {ham_count} messages ({ham_count/len(test_messages)*100:.1f}%)")
print(f"Best Model Used: {best_model_name}")
print(f"Confidence Method: {'Probability' if hasattr(best_model, 'predict_proba') else 'Decision Function (SVM)'}")

print(f"\n" + "="*50)
print("MODEL PERFORMANCE RECAP")
print("="*50)
print(f"Test Accuracy: {accuracy:.1%}")
print(f"Cross-Validation Score: {results[best_model_name]['cv_mean']:.1%}")
print(f"Precision: {precision:.1%}")
print(f"Recall: {recall:.1%}")
print(f"F1-Score: {f1:.1%}")

# 10. RESULTS AND SUMMARY
print("\n10. RESULTS AND SUMMARY")
print("-" * 25)

print(f"✓ Total message count: {len(df)}")
print(f"✓ Best model: {best_model_name}")
print(f"✓ Highest F1-Score: {best_model_info['f1_score']:.4f}")
print(f"✓ Highest accuracy: {best_model_info['accuracy']:.4f}")
print(f"✓ Precision: {best_model_info['precision']:.4f}")
print(f"✓ Recall: {best_model_info['recall']:.4f}")

print("\nProject completed successfully")
print("\nUse 'predict_message()' function to classify new messages.")
print("Example: predict_message('Your message here', best_model, best_vectorizer)")

print("\n Model saving instructions:")
print("import joblib")
print("joblib.dump(best_model, 'spam_classifier_model.pkl')")
print("joblib.dump(best_vectorizer, 'tfidf_vectorizer.pkl')")
print("\n Model loading instructions:")
print("loaded_model = joblib.load('spam_classifier_model.pkl')")
print("loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')")