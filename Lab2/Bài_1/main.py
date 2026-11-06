import numpy as np
from data_process import data_process_logistic
from visual import visualize_loss
from LogisticRegression import LogisticRegression

def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        prec = tp / (tp + fp + 1e-15)
        rec = tp / (tp + fn + 1e-15)
        f1_c = 2 * (prec * rec) / (prec + rec + 1e-15)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1_c)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    accuracy = np.mean(y_true == y_pred) * 100

    return accuracy, macro_precision, macro_recall, macro_f1

def main():
    np.random.seed(42)

    X_train_bin, X_test_bin, y_train_bin, y_test_bin, scaler, features = data_process_logistic(
        data_path='tiki_diengiadung.csv',
        test_size=0.25,
        random_state=42
    )
    # Khởi tạo và huấn luyện mô hình
    lr_model = LogisticRegression(learning_rate=0.1, epochs=500, verbose=True)
    lr_model.fit(X_train_bin, y_train_bin)

    visualize_loss(lr_model.losses, title="Loss Curve - Logistic Regression")

    # In thông tin loss
    print(f"Loss ban đầu: {lr_model.losses[0]:.5f}")
    print(f"Loss sau {lr_model.epochs} epochs: {lr_model.losses[-1]:.5f}")

    # Đánh giá mô hình trên tập test
    y_pred = lr_model.predict(X_test_bin)

    # Tính các chỉ số đánh giá
    acc, prec, rec, f1 = compute_multiclass_metrics(
        y_test_bin,
        y_pred,
        num_classes=2
    )
    
    print(f" - Accuracy:        {acc:.2f}%")
    print(f" - Macro Precision: {prec:.4f}")
    print(f" - Macro Recall:    {rec:.4f}")
    print(f" - Macro F1-Score:  {f1:.4f}")

    print("=" * 45)

if __name__ == '__main__':
    main()