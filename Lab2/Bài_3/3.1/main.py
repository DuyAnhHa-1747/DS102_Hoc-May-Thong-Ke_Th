import os
import numpy as np
from data_process import data_process_logistic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

def main():
    np.random.seed(42)

    X_train, X_test, y_train, y_test, scaler, features = data_process_logistic(
        data_path='tiki_diengiadung.csv', test_size=0.25, random_state=42
    )

    model = LogisticRegression(max_iter=1000, C=10.0, solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_prec = precision_score(y_train, y_train_pred)
    test_prec = precision_score(y_test, y_test_pred)

    train_rec = recall_score(y_train, y_train_pred)
    test_rec = recall_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    train_acc = accuracy_score(y_train, y_train_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100

    print(f"{'Metric':<20} | {'Train Set':<20} | {'Test Set':<20}")
    print("-" * 65)
    print(f"{'Precision':<20} | {train_prec:<20.4f} | {test_prec:<20.4f}")
    print(f"{'Recall':<20} | {train_rec:<20.4f} | {test_rec:<20.4f}")
    print(f"{'F1-Score':<20} | {train_f1:<20.4f} | {test_f1:<20.4f}")
    print(f"{'Accuracy (%)':<20} | {train_acc:<20.2f}% | {test_acc:<20.2f}%")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Gia_Thap (0)', 'Gia_Cao (1)']))

if __name__ == "__main__":
    main()