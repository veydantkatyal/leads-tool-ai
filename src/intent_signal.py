import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import joblib
from imblearn.over_sampling import SMOTE

intent_keywords = [
    "software", "technology", "analytics", "consulting", "services", "cloud", "hosting",
    "marketing", "enterprise", "platform", "solution", "data", "saas", "media", "security"
]
intent_phrases = [
    "expanding to", "seeking investment", "raising capital", "new office", "launching", "hiring for"
]

def detect_intent(text):
    text = str(text).lower()
    keyword_count = sum(kw in text for kw in intent_keywords)
    phrase_present = any(phrase in text for phrase in intent_phrases)
    return int(keyword_count >= 2 or phrase_present)

def prepare_data(input_csv):
    df = pd.read_csv(input_csv, dtype=str)
    df['text_features'] = (
        df['name'].fillna('') + ' ' +
        df['market'].fillna('') + ' ' +
        df['category_list'].fillna('')
    )
    df['intent'] = df['text_features'].apply(detect_intent)
    return df

def train_intent_model(df, max_features=200):
    X_text = df['text_features'].fillna('')
    y = df['intent'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_tfidf, y_train)
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train_bal, y_train_bal)
    return clf, tfidf, X_test_tfidf, y_test

def evaluate_model(clf, X_test_tfidf, y_test):
    y_pred = clf.predict(X_test_tfidf)
    y_pred_proba = clf.predict_proba(X_test_tfidf)[:,1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xticks([0,1], ['No Intent', 'Intent'])
    plt.yticks([0,1], ['No Intent', 'Intent'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.show()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_proba)))
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def save_artifacts(clf, tfidf):
    joblib.dump(clf, 'models/intent_model.pkl')
    joblib.dump(tfidf, 'models/intent_tfidf.pkl')
    print("Intent model and vectorizer saved!")

def example_inference(clf, tfidf, example_text):
    X_example = tfidf.transform([example_text])
    intent_pred = clf.predict(X_example)[0]
    intent_proba = clf.predict_proba(X_example)[0,1]
    print(f"Text: {example_text}")
    print(f"Intent Prediction: {'Intent' if intent_pred else 'No Intent'} (probability: {intent_proba:.2f})")

if __name__ == '__main__':
    input_csv = 'data/leads_recommender.csv'
    df = prepare_data(input_csv)
    clf, tfidf, X_test_tfidf, y_test = train_intent_model(df)
    evaluate_model(clf, X_test_tfidf, y_test)
    save_artifacts(clf, tfidf)
    example_inference(clf, tfidf, "Our new cloud platform offers advanced analytics and security.") 