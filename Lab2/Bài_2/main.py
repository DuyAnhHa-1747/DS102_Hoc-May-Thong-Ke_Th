import numpy as np
from SoftmaxRegression import SoftmaxRegression
from data_process import data_process_softmax
from visual import visualize_loss

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

    X_train, X_test, y_train, y_test, num_classes, class_names, features = data_process_softmax(
        data_path='tiki_diengiadung.csv',
        test_size=0.25,
        random_state=42
    )

    print(f"Số lượng lớp phân loại: {num_classes} {list(class_names)}")
    print(f"Kích thước tập Train: {X_train.shape}")
    print(f"Kích thước tập Test:  {X_test.shape}\n")

    # Khởi tạo và huấn luyện mô hình
    model = SoftmaxRegression(learning_rate=0.1, epochs=500, num_classes=num_classes)
    model.fit(X_train, y_train)

    visualize_loss(model.losses, title="Loss Curve - Softmax Regression")

    # In thông tin loss
    print(f"Loss ban đầu: {model.losses[0]:.5f}")
    print(f"Loss sau {model.epochs} epochs: {model.losses[-1]:.5f}\n")

    # Đánh giá mô hình trên tập test
    y_pred = model.predict(X_test)

    # Tính các chỉ số đánh giá
    acc, prec, rec, f1 = compute_multiclass_metrics(
        y_test,
        y_pred,
        num_classes=num_classes
    )

    print(f" - Accuracy:        {acc:.2f}%")
    print(f" - Macro Precision: {prec:.4f}")
    print(f" - Macro Recall:    {rec:.4f}")
    print(f" - Macro F1-Score:  {f1:.4f}")

    print("=" * 45)

if __name__ == '__main__':
    main()