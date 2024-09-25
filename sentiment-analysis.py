import pandas as pd
import numpy as np
import re
import stanza
from pathlib import Path

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.utils import compute_sample_weight
from tabulate import tabulate
import xgboost as xgb
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import seaborn as sns
import matplotlib.pyplot as plt

# stanza.download('sr')

# Initialize Stanza pipeline
nlp = stanza.Pipeline('sr', processors='tokenize,pos,lemma', tokenize_pretokenized=False, verbose=False)

# Serbian stopwords
serbian_stopwords = set(Path('C:/Users/Barbara/PycharmProjects/booking-sentiment-analysis/serbian_stopwords.txt').read_text(encoding='utf-8').splitlines())

def check_data(data):
    print("About dataset")
    print("--------------")
    print(data.info())

    print("\nDimensions")
    print("----------")
    print(data.shape)

    print("\nTypes")
    print("-------------------")
    print(data.dtypes)

    print("\nMissing values")
    print("--------------------")
    print(data.isnull().sum())

    print("\nDuplicates")
    print("----------")
    print(data.duplicated().sum())
    # duplicates = data[data.duplicated(keep=False)]

    print("\nDescription")
    print("-----------")
    print(data.describe().T)

    print("\nExample")
    print("-------")
    pd.set_option('display.max_columns', None)
    print(data.head(10))


def preprocess_data(data):
    data_cleaned = data.drop_duplicates().copy()
    data_cleaned['comments'] = data_cleaned['positive'].fillna('') + " " + data_cleaned['negative'].fillna('')
    return data_cleaned


def preprocess_comments(comments):
    comments = comments.lower() # lowercase
    comments = re.sub(r'[^A-Za-zčćžšđČĆŽŠĐ\s]', '', comments)  # Keeps only alphabetic characters and spaces
    comments = re.sub(r'\d+', '', comments) # remove numbers

    doc = nlp(comments)
    lemmatized_tokens = [word.lemma for sentence in doc.sentences for word in sentence.words]
    lemmatized_tokens = [token for token in lemmatized_tokens if token not in serbian_stopwords]

    # Remove extra whitespaces
    cleaned_text = ' '.join(lemmatized_tokens)
    cleaned_text = cleaned_text.strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text


def booking_sentiment_analysis_non_balanced(data):
    print("Use non balanced dataset")
    # Work with non balanced data
    apply_standard_classification_algorithms(data)
    # apply_big_language_models(data)


def booking_sentiment_analysis_balanced(data):
    print("Use balanced dataset")

    # Determine number of instances in each category
    print("Instances per category [before balancing]")
    print(data['category'].value_counts())

    X = data.drop('category', axis=1)  # Features
    y = data['category']  # Target

    under_sampler = RandomUnderSampler(sampling_strategy={
        'pozitivan': 109  # Target count for the majority class
    }, random_state=42)

    # Undersample the majority class
    X_under, y_under = under_sampler.fit_resample(X, y)

    under_sampled_data = pd.DataFrame(X_under, columns=X.columns)
    under_sampled_data['category'] = y_under

    over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)

    # Oversample the minority classes
    X_balanced, y_balanced = over_sampler.fit_resample(under_sampled_data.drop('category', axis=1),
                                                       under_sampled_data['category'])
    balanced_data = pd.DataFrame(X_balanced, columns=X_balanced.columns)
    balanced_data['category'] = y_balanced

    print("Instances per category [after balancing]")
    print(balanced_data['category'].value_counts())
    balanced_data.to_csv('serbia-comments-balanced-sampled.csv', index=False, encoding='utf-8')

    # Work with balanced_train_data
    apply_standard_classification_algorithms(balanced_data)
    # apply_big_language_models(balanced_data)


def apply_standard_classification_algorithms(data):
    print("Use standard classification algorithms: Logistic regression, Naive Bayes, SVM ...")
    data['comments_preprocessed'] = data['comments'].apply(preprocess_comments)
    cleaned_data = data[data['comments_preprocessed'].notna() & (data['comments_preprocessed'].str.strip() != '')]
    # cleaned_data.to_csv('comments-preprocessed-non-balanced.csv', index=False, encoding='utf-8')
    cleaned_data.to_csv('comments-preprocessed-balanced.csv', index=False, encoding='utf-8') # undersample

    # data = pd.read_csv("C:/Users/Barbara/PycharmProjects/booking-sentiment-analysis/comments-preprocessed-non-balanced.csv", encoding='utf-8')
    # data = pd.read_csv("C:/Users/Barbara/PycharmProjects/booking-sentiment-analysis/comments-preprocessed-balanced.csv", encoding='utf-8')

    X = cleaned_data['comments_preprocessed'] # Features

    le = LabelEncoder()
    y_encoded = le.fit_transform(cleaned_data['category']) # Target variable encoded

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Initialize TF-IDF vectorizer
    TFIDF_vect= TfidfVectorizer(max_features=2000)
    vectorizer = CountVectorizer(max_features=2000)

    X_train_TFIDF = TFIDF_vect.fit_transform(X_train)
    X_test_TFIDF = TFIDF_vect.transform(X_test)

    X_train_count = vectorizer.fit_transform(X_train)
    X_test_count = vectorizer.transform(X_test)

    results = []
    results_count = []
    # ----------------------------------
    # Logistic regression
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_train_TFIDF, y_train)
    y_pred_lr = lr.predict(X_test_TFIDF)

    test_acc_lr = round(accuracy_score(y_test, y_pred_lr) * 100, 2)
    class_report_lr = classification_report(y_test, y_pred_lr, target_names=le.classes_, zero_division=0)
    results.append({
        'Model': 'Logistic Regression',
        'Accuracy': test_acc_lr,
        'Report': class_report_lr
    })
    # ----------------------------------
    lr.fit(X_train_count, y_train)
    y_pred_lr = lr.predict(X_test_count)

    test_acc_lr = round(accuracy_score(y_test, y_pred_lr) * 100, 2)
    class_report_lr = classification_report(y_test, y_pred_lr, target_names=le.classes_, zero_division=0)
    results_count.append({
        'Model': 'Logistic Regression',
        'Accuracy': test_acc_lr,
        'Report': class_report_lr
    })
    # ----------------------------------
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_TFIDF, y_train)
    y_pred_nb = nb.predict(X_test_TFIDF)

    test_acc_nb = round(accuracy_score(y_test, y_pred_nb) * 100, 2)
    class_report_nb = classification_report(y_test, y_pred_nb, target_names=le.classes_, zero_division=0)
    results.append({
        'Model': 'Naive Bayes',
        'Accuracy': test_acc_nb,
        'Report': class_report_nb
    })
    # ----------------------------------
    nb.fit(X_train_count, y_train)
    y_pred_nb = nb.predict(X_test_count)

    test_acc_nb = round(accuracy_score(y_test, y_pred_nb) * 100, 2)
    class_report_nb = classification_report(y_test, y_pred_nb, target_names=le.classes_, zero_division=0)
    results_count.append({
        'Model': 'Naive Bayes',
        'Accuracy': test_acc_nb,
        'Report': class_report_nb
    })
    # ----------------------------------
    # Support Vector Machine
    svm = SVC(kernel='linear', class_weight='balanced')
    svm.fit(X_train_TFIDF, y_train)
    y_pred_svm = svm.predict(X_test_TFIDF)

    test_acc_svm = round(accuracy_score(y_test, y_pred_svm) * 100, 2)
    class_report_svm = classification_report(y_test, y_pred_svm, target_names=le.classes_, zero_division=0)
    results.append({
        'Model': 'Support Vector Machine (SVM)',
        'Accuracy': test_acc_svm,
        'Report': class_report_svm
    })
    # ----------------------------------
    svm.fit(X_train_count, y_train)
    y_pred_svm = svm.predict(X_test_count)

    test_acc_svm = round(accuracy_score(y_test, y_pred_svm) * 100, 2)
    class_report_svm = classification_report(y_test, y_pred_svm, target_names=le.classes_, zero_division=0)
    results_count.append({
        'Model': 'Support Vector Machine (SVM)',
        'Accuracy': test_acc_svm,
        'Report': class_report_svm
    })
    # ----------------------------------
    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    rf.fit(X_train_TFIDF, y_train)
    y_pred_rf = rf.predict(X_test_TFIDF)

    test_acc_rf = round(accuracy_score(y_test, y_pred_rf) * 100, 2)
    class_report_rf = classification_report(y_test, y_pred_rf, target_names=le.classes_, zero_division=0)
    results.append({
        'Model': 'Random Forest',
        'Accuracy': test_acc_rf,
        'Report': class_report_rf
    })
    # ----------------------------------
    rf.fit(X_train_count, y_train)
    y_pred_rf = rf.predict(X_test_count)

    test_acc_rf = round(accuracy_score(y_test, y_pred_rf) * 100, 2)
    class_report_rf = classification_report(y_test, y_pred_rf, target_names=le.classes_, zero_division=0)
    results_count.append({
        'Model': 'Random Forest',
        'Accuracy': test_acc_rf,
        'Report': class_report_rf
    })
    # ----------------------------------
    # XGB
    dtrain = xgb.DMatrix(X_train_TFIDF, label=y_train)
    dtest = xgb.DMatrix(X_test_TFIDF)

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    dtrain.set_weight(sample_weights)

    params = {'objective': 'multi:softmax', 'num_class': 3, 'max_depth': 5, 'learning_rate': 0.1}
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)
    y_pred_xgb = xgb_model.predict(dtest)

    test_acc_xgb = round(accuracy_score(y_test, y_pred_xgb) * 100, 2)
    class_report_xgb = classification_report(y_test, y_pred_xgb, target_names=le.classes_, zero_division=0)
    results.append({
        'Model': 'XGB',
        'Accuracy': test_acc_xgb,
        'Report': class_report_xgb
    })
    # ----------------------------------
    dtrain = xgb.DMatrix(X_train_count, label=y_train)
    dtest = xgb.DMatrix(X_test_count)
    params = {'objective': 'multi:softmax', 'num_class': 3, 'max_depth': 5, 'learning_rate': 0.1}
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)
    y_pred_xgb = xgb_model.predict(dtest)

    test_acc_xgb = round(accuracy_score(y_test, y_pred_xgb) * 100, 2)
    class_report_xgb = classification_report(y_test, y_pred_xgb, target_names=le.classes_, zero_division=0)
    results_count.append({
        'Model': 'XGB',
        'Accuracy': test_acc_xgb,
        'Report': class_report_xgb
    })
    # ----------------------------------

    # Print the results
    print("===TF-IDF===")
    for result in results:
        print("=" * 50)
        print(f"Model: {result['Model']}")
        print(f"Accuracy: {result['Accuracy']}%")
        print("Classification Report:")
        print(result['Report'])
        print("=" * 50)

    # Summary table for accuracies
    summary = [(result['Model'], f"{result['Accuracy']}%") for result in results]
    print("\nSummary of Model Accuracies (TF-IDF):")
    print(tabulate(summary, headers=["Model", "Accuracy"], tablefmt="pretty"))
    # ----------------------------------
    print("===CountVectorizer===")
    for result in results_count:
        print("=" * 50)
        print(f"Model: {result['Model']}")
        print(f"Accuracy: {result['Accuracy']}%")
        print("Classification Report:")
        print(result['Report'])
        print("=" * 50)

    # Summary table for accuracies
    summary = [(result['Model'], f"{result['Accuracy']}%") for result in results_count]
    print("\nSummary of Model Accuracies (CountVectorizer):")
    print(tabulate(summary, headers=["Model", "Accuracy"], tablefmt="pretty"))


def apply_big_language_models(data):
    print("Apply XLM-RoBERTa language model")

    xlm_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=3)

    X = data['comments']
    y = data['category'].apply(lambda x: 0 if x == 'negativan' else 1 if x == 'neutralan' else 2)  # Encode categories
    # 0 - negativan, 1 - neutralan, 2 - pozitivan

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_encodings = xlm_tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
    test_encodings = xlm_tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = SentimentDataset(train_encodings, y_train.tolist())
    test_dataset = SentimentDataset(test_encodings, y_test.tolist())

    training_args = TrainingArguments(
        output_dir='./xlm-roberta-results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Load the xlm-roberta trained model
    # model = AutoModelForSequenceClassification.from_pretrained('./xlm-roberta-results/checkpoint-36')

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, output_dict=False, zero_division=0)
        print("Classification Report:")
        print(report)
        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:")
        print(cm)
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('Actual labels')
        plt.title('Confusion Matrix')
        plt.show()
        return {'eval_accuracy': round(acc * 100, 2),
                'eval_precision': round(precision * 100, 2),
                'eval_recall': round(recall * 100, 2),
                'eval_f1': round(f1 * 100, 2)}

    trainer.compute_metrics = compute_metrics

    trainer.train()
    eval_results = trainer.evaluate()

    print("Evaluation Metrics (%):")
    print("Accuracy:", eval_results["eval_accuracy"])
    print("Precision:", eval_results["eval_precision"])
    print("Recall:", eval_results["eval_recall"])
    print("F1 Score:", eval_results["eval_f1"])


def booking_sentiment_analysis():
    booking_data = pd.read_csv("C:/Users/Barbara/Downloads/MAS/NLP/serbia-comments.csv", encoding='utf-8')
    check_data(booking_data)
    booking_data_cleaned = preprocess_data(booking_data)

    conditions = [
        (booking_data_cleaned['rating'] >= 1) & (booking_data_cleaned['rating'] <= 3),
        (booking_data_cleaned['rating'] >= 4) & (booking_data_cleaned['rating'] <= 6),
        (booking_data_cleaned['rating'] >= 7) & (booking_data_cleaned['rating'] <= 10)
    ]
    categories = ['negativan', 'neutralan', 'pozitivan']

    booking_data_cleaned['category'] = np.select(conditions, categories, default='unknown')
    # print(booking_data_cleaned.head())

    # booking_data_cleaned.to_csv('serbia-comments-cleaned.csv', index=False, encoding='utf-8')

    # booking_sentiment_analysis_non_balanced(booking_data_cleaned)
    booking_sentiment_analysis_balanced(booking_data_cleaned)


if __name__ == '__main__':
    booking_sentiment_analysis()