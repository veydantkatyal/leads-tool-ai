import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def preprocess_data(input_csv):
    df = pd.read_csv(input_csv)
    current_year = datetime.now().year
    df['company_age'] = current_year - df['founded_year'].fillna(current_year).astype(int)
    df['last_funding_at'] = pd.to_datetime(df['last_funding_at'], errors='coerce')
    df['time_since_last_funding'] = (pd.Timestamp.now() - df['last_funding_at']).dt.days / 365.25
    df['time_since_last_funding'] = df['time_since_last_funding'].fillna(df['time_since_last_funding'].median())
    df['founded_at'] = pd.to_datetime(df['founded_at'], errors='coerce')
    df['first_funding_at'] = pd.to_datetime(df['first_funding_at'], errors='coerce')
    df['time_to_first_funding'] = (df['first_funding_at'] - df['founded_at']).dt.days / 365.25
    df['time_to_first_funding'] = df['time_to_first_funding'].fillna(df['time_to_first_funding'].median())
    def group_rare(series, min_count=100):
        freq = series.value_counts()
        return series.apply(lambda x: x if freq[x] >= min_count else 'other')
    for col in ['city', 'market']:
        if col in df.columns:
            df[col] = group_rare(df[col].fillna(''), min_count=100)
    df['text_features'] = df['category_list'].fillna('') + ' ' + df['market'].fillna('')
    return df

def extract_features(df, max_tfidf_features=100):
    tfidf = TfidfVectorizer(max_features=max_tfidf_features)
    tfidf_matrix = tfidf.fit_transform(df['text_features'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    funding_cols = [
        'seed','venture','equity_crowdfunding','undisclosed','convertible_note','debt_financing',
        'angel','grant','private_equity','post_ipo_equity','post_ipo_debt','secondary_market',
        'product_crowdfunding','round_a','round_b','round_c','round_d','round_e','round_f','round_g','round_h'
    ]
    funding_cols = [col for col in funding_cols if col in df.columns]
    features = [
        'market', 'category_list', 'funding_rounds', 'status',
        'country_code', 'state_code', 'region', 'city', 'company_age',
        'time_since_last_funding', 'time_to_first_funding'
    ] + funding_cols
    X = df[features].copy()
    y = df['converted']
    cat_features = ['market', 'category_list', 'status', 'country_code', 'state_code', 'region', 'city']
    for col in cat_features:
        X[col] = X[col].fillna('').astype(str)
    for col in X.columns:
        if col not in cat_features:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_cat = ohe.fit_transform(X[cat_features])
    X_num = X.drop(columns=cat_features).astype(float).values
    X_all = np.hstack([X_cat, X_num, tfidf_df.values])
    return X_all, y, ohe, tfidf, X, tfidf_df

def train_model(X_all, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=500,  
        max_depth=15,      
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, X, tfidf_df):
    y_pred_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_proba)))
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    plt.figure(figsize=(4,3))
    sns.countplot(x=y_test)
    plt.title('Class Distribution (Converted)')
    plt.show()
    importances = model.feature_importances_
    feature_names = list(X.columns) + list(tfidf_df.columns)
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(8,6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def save_artifacts(model, ohe, tfidf, X, tfidf_df, y):
    joblib.dump(model, 'models/lead_xgb_model.pkl')
    joblib.dump(ohe, 'models/lead_ohe.pkl')
    joblib.dump(tfidf, 'models/lead_tfidf.pkl')
    tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_df.shape[1])]
    tfidf_features = pd.DataFrame(tfidf_df.values, columns=tfidf_feature_names)
    preprocessed_df = pd.concat([X.reset_index(drop=True), tfidf_features], axis=1)
    preprocessed_df['converted'] = y.values
    preprocessed_df.to_csv('data/lead_scoring.csv', index=False)
    print("Artifacts and preprocessed dataset saved.")

def example_inference(model, ohe, tfidf, X, tfidf_df):
    cat_features = ['market', 'category_list', 'status', 'country_code', 'state_code', 'region', 'city']
    X_new_cat = ohe.transform(X[cat_features].iloc[:5])
    X_new_num = X.drop(columns=cat_features).iloc[:5].astype(float).values
    X_new_tfidf = tfidf.transform(X['text_features'].iloc[:5]).toarray()
    X_new = np.hstack([X_new_cat, X_new_num, X_new_tfidf])
    probs = model.predict_proba(X_new)[:,1]
    lead_scores = (probs * 100).round(2)
    print("Lead scores (0-100):", lead_scores)

def main():
    df = preprocess_data('data/leads_cleaned.csv')
    X_all, y, ohe, tfidf, X, tfidf_df = extract_features(df)
    tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_df.shape[1])]
    tfidf_features = pd.DataFrame(tfidf_df.values, columns=tfidf_feature_names)
    preprocessed_df = pd.concat([X.reset_index(drop=True), tfidf_features], axis=1)
    preprocessed_df['converted'] = y.values
    preprocessed_df.to_csv('data/lead_scoring.csv', index=False)
    model, X_train, X_test, y_train, y_test = train_model(X_all, y)
    evaluate_model(model, X_test, y_test, X, tfidf_df)
    save_artifacts(model, ohe, tfidf, X, tfidf_df, y)
    example_inference(model, ohe, tfidf, X, tfidf_df)

if __name__ == '__main__':
    main() 