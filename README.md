# Нейронная сеть для классификации рукописных цифр

## Описание проекта

Данный проект представляет собой реализацию нейронной сети для классификации рукописных цифр из набора данных MNIST. Проект включает в себя полный цикл работы с моделью: от подготовки данных до обучения и тестирования.

## Структура проекта

```
├── lab1.py             # Файл для работы с моделью
├── requirements.txt    # Зависимости проекта
```

## Установка и настройка

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

## Использование

### Обучение модели 

```bash
python lab1.py train --train_dir mnist_images/training --test_dir mnist_images/test --epochs 20 --lr 0.01
```

Параметры:

- `train`: режим запуска
- `--train_dir`: путь до обучающих данных
- `--test_dir`: путь до тестовых данных
- `--epochs`: количество эпох обучения
- `--lr`: скорость обучения

Для текущих параметров вывод будет выглядеть примерно так:

```
Загрузка тренировочных данных...
Загрузка тестовых данных...
Начало обучения...

Epoch 1/20
Train Loss: 0.4591
Accuracy: 0.8689, Precision: 0.8684, Recall: 0.8672, F1-Score: 0.8671

Epoch 2/20
Train Loss: 0.2019
Accuracy: 0.9423, Precision: 0.9418, Recall: 0.9417, F1-Score: 0.9417

Epoch 3/20
Train Loss: 0.1492
Accuracy: 0.9568, Precision: 0.9565, Recall: 0.9564, F1-Score: 0.9564

Epoch 4/20
Train Loss: 0.1180
Accuracy: 0.9662, Precision: 0.9660, Recall: 0.9659, F1-Score: 0.9659

Epoch 5/20
Train Loss: 0.0974
Accuracy: 0.9725, Precision: 0.9723, Recall: 0.9722, F1-Score: 0.9722

Epoch 6/20
Train Loss: 0.0821
Accuracy: 0.9767, Precision: 0.9765, Recall: 0.9765, F1-Score: 0.9765

Epoch 7/20
Train Loss: 0.0690
Accuracy: 0.9803, Precision: 0.9802, Recall: 0.9802, F1-Score: 0.9802

Epoch 8/20
Train Loss: 0.0604
Accuracy: 0.9828, Precision: 0.9828, Recall: 0.9827, F1-Score: 0.9827

Epoch 9/20
Train Loss: 0.0531
Accuracy: 0.9847, Precision: 0.9846, Recall: 0.9846, F1-Score: 0.9846

Epoch 10/20
Train Loss: 0.0468
Accuracy: 0.9867, Precision: 0.9866, Recall: 0.9866, F1-Score: 0.9866

Epoch 11/20
Train Loss: 0.0409
Accuracy: 0.9886, Precision: 0.9886, Recall: 0.9886, F1-Score: 0.9886

Epoch 12/20
Train Loss: 0.0361
Accuracy: 0.9904, Precision: 0.9904, Recall: 0.9904, F1-Score: 0.9904

Epoch 13/20
Train Loss: 0.0319
Accuracy: 0.9915, Precision: 0.9915, Recall: 0.9915, F1-Score: 0.9915

Epoch 14/20
Train Loss: 0.0284
Accuracy: 0.9927, Precision: 0.9927, Recall: 0.9927, F1-Score: 0.9927

Epoch 15/20
Train Loss: 0.0244
Accuracy: 0.9941, Precision: 0.9941, Recall: 0.9941, F1-Score: 0.9941

Epoch 16/20
Train Loss: 0.0217
Accuracy: 0.9950, Precision: 0.9950, Recall: 0.9950, F1-Score: 0.9950

Epoch 17/20
Train Loss: 0.0195
Accuracy: 0.9956, Precision: 0.9956, Recall: 0.9956, F1-Score: 0.9956

Epoch 18/20
Train Loss: 0.0175
Accuracy: 0.9961, Precision: 0.9961, Recall: 0.9961, F1-Score: 0.9961

Epoch 19/20
Train Loss: 0.0152
Accuracy: 0.9971, Precision: 0.9971, Recall: 0.9971, F1-Score: 0.9971

Epoch 20/20
Train Loss: 0.0133
Accuracy: 0.9977, Precision: 0.9977, Recall: 0.9977, F1-Score: 0.9977

Test Metrics:
Accuracy: 0.9813, Precision: 0.9811, Recall: 0.9811, F1-Score: 0.9811

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.98      0.98      1032
           3       0.98      0.97      0.98      1010
           4       0.98      0.98      0.98       982
           5       0.97      0.98      0.98       892
           6       0.99      0.98      0.98       958
           7       0.98      0.98      0.98      1028
           8       0.98      0.97      0.98       974
           9       0.98      0.98      0.98      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000

Веса сохранены в weights.npz
```

### Тестирование модели 

```bash
python lab1.py test --test_dir mnist_images/test --weights_file weights.npz
```

Параметры:

- `test`: режим запуска
- `--test_dir`: путь к директории с тестовыми данными
- `--weights_file`: путь к сохраненным весам модели

Для текущих параметров вывод будет выглядеть примерно так:

```
Загрузка тестовых данных из mnist_images/test...

Test Metrics:
Accuracy: 0.9813, Precision: 0.9811, Recall: 0.9811, F1-Score: 0.9811

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.98      0.98      1032
           3       0.98      0.97      0.98      1010
           4       0.98      0.98      0.98       982
           5       0.97      0.98      0.98       892
           6       0.99      0.98      0.98       958
           7       0.98      0.98      0.98      1028
           8       0.98      0.97      0.98       974
           9       0.98      0.98      0.98      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000
```

### Предсказание на отдельном изображении 

```bash
python lab1.py predict --image_path mnist_images/test/0/img_3.png --weights_file weights.npz
```

Параметры:

- `predict`: режим запуска
- `--image_path`: путь к директории с тестовым изображением
- `--weights_file`: путь к сохраненным весам модели

Для текущих параметров вывод будет выглядеть примерно так:

```
Предсказание: Цифра — 0
Confidence: 100.00%
```

## Требования к датасету

Датасет должен иметь следующую структуру:

```
dataset/
├── training/
│   ├── 0/    # Изображения 0 для обучения
│   ├── 1/    # Изображения 1 для обучения
│   ├── 2/    # Изображения 2 для обучения
│   ├── 3/    # Изображения 3 для обучения
│   ├── 4/    # Изображения 4 для обучения
│   ├── 5/    # Изображения 5 для обучения
│   ├── 6/    # Изображения 6 для обучения
│   ├── 7/    # Изображения 7 для обучения
│   ├── 8/    # Изображения 8 для обучения
│   └── 9/    # Изображения 9 для обучения
└── test/
    ├── 0/    # Изображения 0 для обучения
    ├── 1/    # Изображения 1 для обучения
    ├── 2/    # Изображения 2 для обучения
    ├── 3/    # Изображения 3 для обучения
    ├── 4/    # Изображения 4 для обучения
    ├── 5/    # Изображения 5 для обучения
    ├── 6/    # Изображения 6 для обучения
    ├── 7/    # Изображения 7 для обучения
    ├── 8/    # Изображения 8 для обучения
    └── 9/    # Изображения 9 для обучения
```

Требования к изображениям:

- Формат: одноканальные изображения (оттенки серого)

Для генерации собственного датасета используйте скрипт generate.py:

```bash
python generate.py
python lab1.py generate --output_dir dataset --format .png --train_size 10000 --test_size 2000
```

Параметры:

- `--output_dir`: директория для сохранения сгенерированного датасета
- `--format`: формат изображений
- `--train_size`: количество обучающих изображений
- `--test_size`: количество тестовых изображений

## Метрики оценки

При тестировании модели вычисляются следующие метрики:

- Accuracy (точность)
- Precision (точность по классам)
- Recall (полнота)
- F1-score (F1-мера)
