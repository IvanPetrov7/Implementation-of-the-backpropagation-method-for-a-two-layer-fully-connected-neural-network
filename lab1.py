import numpy as np
import os
import sys
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# -----------------------------
# Гиперпараметры
# -----------------------------
learning_rate = 0.01
momentum = 0.9
batch_size = 128
epochs = 20
WEIGHTS_FILE = "weights.npz"

# -----------------------------
# Активационные функции и их производные
# -----------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# -----------------------------
# Кросс-энтропия
# -----------------------------
def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# -----------------------------
# Функция для чтения изображений из директорий
# -----------------------------
def read_images(directory, image_size=(28, 28)):
    data = []
    labels = []

    for digit in range(10):
        digit_dir = os.path.join(directory, str(digit))
        if not os.path.exists(digit_dir):
            print(f"Директория не найдена: {digit_dir}")
            continue

        for filename in os.listdir(digit_dir):
            filepath = os.path.join(digit_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Не удалось загрузить изображение: {filepath}")
                continue

            img = cv2.resize(img, image_size)
            img = img.astype(np.float32) / 255.0
            data.append(img.flatten())  # Преобразование в вектор 784
            labels.append(digit)

    data = np.array(data)
    labels = np.array(labels).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    labels_onehot = encoder.fit_transform(labels)

    return data, labels_onehot

# -----------------------------
# Инициализация весов и смещений
# -----------------------------
def init_weights(input_size, hidden1_size, hidden2_size, output_size):
    scale = np.sqrt(2.0 / (input_size + hidden1_size))
    W1 = np.random.randn(input_size, hidden1_size) * scale
    b1 = np.zeros((1, hidden1_size))

    scale = np.sqrt(2.0 / (hidden1_size + hidden2_size))
    W2 = np.random.randn(hidden1_size, hidden2_size) * scale
    b2 = np.zeros((1, hidden2_size))

    scale = np.sqrt(2.0 / (hidden2_size + output_size))
    W3 = np.random.randn(hidden2_size, output_size) * scale
    b3 = np.zeros((1, output_size))

    return W1, b1, W2, b2, W3, b3

# -----------------------------
# Forward pass
# -----------------------------
def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    cache = (Z1, A1, Z2, A2, Z3, A3)
    return A3, cache

# -----------------------------
# Backward pass
# -----------------------------
def backward(X, Y, cache, W1, W2, W3):
    Z1, A1, Z2, A2, Z3, A3 = cache
    m = X.shape[0]

    dZ3 = A3 - Y
    dW3 = A2.T @ dZ3 / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

# -----------------------------
# Обновление весов с моментом
# -----------------------------
def update_weights_momentum(W1, b1, W2, b2, W3, b3,
                            dW1, db1, dW2, db2, dW3, db3,
                            v_W1, v_b1, v_W2, v_b2, v_W3, v_b3):
    v_W1 = momentum * v_W1 - learning_rate * dW1
    v_b1 = momentum * v_b1 - learning_rate * db1

    v_W2 = momentum * v_W2 - learning_rate * dW2
    v_b2 = momentum * v_b2 - learning_rate * db2

    v_W3 = momentum * v_W3 - learning_rate * dW3
    v_b3 = momentum * v_b3 - learning_rate * db3

    W1 += v_W1
    b1 += v_b1
    W2 += v_W2
    b2 += v_b2
    W3 += v_W3
    b3 += v_b3

    return W1, b1, W2, b2, W3, b3, v_W1, v_b1, v_W2, v_b2, v_W3, v_b3

# -----------------------------
# Функция для вычисления метрик
# -----------------------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, prec, rec, f1

# -----------------------------
# Функция обучения
# -----------------------------
def train(train_data, train_labels, test_data, test_labels, total_epochs):
    input_size = 784
    hidden1_size = 256
    hidden2_size = 128
    output_size = 10

    W1, b1, W2, b2, W3, b3 = init_weights(input_size, hidden1_size, hidden2_size, output_size)

    # Инициализация скоростей для момента
    v_W1, v_b1 = np.zeros_like(W1), np.zeros_like(b1)
    v_W2, v_b2 = np.zeros_like(W2), np.zeros_like(b2)
    v_W3, v_b3 = np.zeros_like(W3), np.zeros_like(b3)

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        indices = np.random.permutation(len(train_data))
        loss_total = 0

        all_true = []
        all_pred = []

        for i in range(0, len(train_data), batch_size):
            idx = indices[i:i+batch_size]
            X_batch = train_data[idx]
            Y_batch = train_labels[idx]

            # Forward
            A3, cache = forward(X_batch, W1, b1, W2, b2, W3, b3)

            # Loss
            loss = categorical_crossentropy(Y_batch, A3)
            loss_total += loss * len(X_batch)

            # Accuracy, Precision, Recall, F1
            preds = np.argmax(A3, axis=1)
            true = np.argmax(Y_batch, axis=1)

            all_true.extend(true)
            all_pred.extend(preds)

            # Backward
            dW1, db1, dW2, db2, dW3, db3 = backward(X_batch, Y_batch, cache, W1, W2, W3)

            # Update
            W1, b1, W2, b2, W3, b3, v_W1, v_b1, v_W2, v_b2, v_W3, v_b3 = update_weights_momentum(
                W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3,
                v_W1, v_b1, v_W2, v_b2, v_W3, v_b3
            )

        # Метрики на трейне
        acc, prec, rec, f1 = compute_metrics(all_true, all_pred)
        avg_loss = loss_total / len(train_data)

        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")

    # -----------------------------
    # Оценка на тестовых данных
    # -----------------------------
    A3_test, _ = forward(test_data, W1, b1, W2, b2, W3, b3)
    preds_test = np.argmax(A3_test, axis=1)
    true_test = np.argmax(test_labels, axis=1)

    test_acc, test_prec, test_rec, test_f1 = compute_metrics(true_test, preds_test)

    print("\nTest Metrics:")
    print(f"Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1-Score: {test_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_test, preds_test))

    # Сохранение весов
    np.savez(WEIGHTS_FILE, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    print(f"Веса сохранены в {WEIGHTS_FILE}")

def predict_image(image_path, weights_file):
    if not os.path.exists(weights_file):
        print(f"Файл весов не найден: {weights_file}")
        return

    weights = np.load(weights_file)
    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']
    W3, b3 = weights['W3'], weights['b3']

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Ошибка: не удалось загрузить изображение.")
        return

    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0
    img_flat = img.flatten().reshape(1, -1)

    A3, _ = forward(img_flat, W1, b1, W2, b2, W3, b3)
    prediction = np.argmax(A3, axis=1)[0]
    print(f"Предсказание: Цифра — {prediction}")

# -----------------------------
# Тестирование на всём датасете
# -----------------------------
def test_dataset(test_dir, weights_file):
    if not os.path.exists(weights_file):
        print(f"Файл весов не найден: {weights_file}")
        return

    # Загрузка весов
    weights = np.load(weights_file)
    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']
    W3, b3 = weights['W3'], weights['b3']

    # Загрузка тестовых данных
    print(f"Загрузка тестовых данных из {test_dir}...")
    test_data, test_labels = read_images(test_dir)

    # Прямой проход
    A3_test, _ = forward(test_data, W1, b1, W2, b2, W3, b3)
    preds_test = np.argmax(A3_test, axis=1)
    true_test = np.argmax(test_labels, axis=1)

    # Метрики
    test_acc, test_prec, test_rec, test_f1 = compute_metrics(true_test, preds_test)
    print("\nTest Metrics:")
    print(f"Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1-Score: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_test, preds_test))

# -----------------------------
# Точка входа программы
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("Обучение: python script.py train <train_dir> <test_dir> <epochs> <lr>")
        print("Тестирование: python script.py test <test_dir> <weights_file>")
        print("Прогноз: python script.py predict <image_path> <weights_file>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "train":
        if len(sys.argv) != 6:
            print("Использование: python script.py train <train_dir> <test_dir> <epochs> <lr>")
            sys.exit(1)

        train_dir = sys.argv[2]
        test_dir = sys.argv[3]
        total_epochs = int(sys.argv[4])
        global learning_rate
        learning_rate = float(sys.argv[5])

        print("Загрузка тренировочных данных...")
        train_data, train_labels = read_images(train_dir)
        print("Загрузка тестовых данных...")
        test_data, test_labels = read_images(test_dir)

        print("Начало обучения...")
        train(train_data, train_labels, test_data, test_labels, total_epochs)

    elif mode == "test":
        if len(sys.argv) != 4:
            print("Использование: python script.py test <test_dir> <weights_file>")
            sys.exit(1)

        test_dir = sys.argv[2]
        weights_file = sys.argv[3]
        test_dataset(test_dir, weights_file)

    elif mode == "predict":
        if len(sys.argv) != 4:
            print("Использование: python script.py predict <image_path> <weights_file>")
            sys.exit(1)

        image_path = sys.argv[2]
        weights_file = sys.argv[3]
        predict_image(image_path, weights_file)

    else:
        print("Режим не распознан. Используйте 'train', 'test' или 'predict'.")
        sys.exit(1)

if __name__ == "__main__":
    main()