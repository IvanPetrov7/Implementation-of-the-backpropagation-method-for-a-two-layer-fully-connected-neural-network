# Нейронная сеть для классификации рукописных цифр

## Описание проекта

Данный проект представляет собой реализацию нейронной сети DenseNet для классификации рукописных цифр из набора данных MNIST. Проект включает в себя полный цикл работы с моделью: от подготовки данных до обучения и тестирования.

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

### Обучение модели (train.py)

```bash
python train.py --data_path mnist_data \
                --input_dim 784 \
                --hidden_layers 1024 \
                --output_dim 10 \
                --learning_rate 0.1 \
                --epochs 50 \
                --weights_path model_weights.npz
```

Параметры:

- `--data_path`: путь к директории с обучающими данными
- `--input_dim`: размерность входных данных
- `--hidden_layers`: размеры скрытых слоев
- `--output_dim`: размерность выходного слоя
- `--learning_rate`: скорость обучения
- `--epochs`: количество эпох обучения
- `--weights_path`: путь для сохранения весов модели

Для текущих параметров вывод будет выглядеть примерно так:

```
Number of training samples: 60000
Training model...
Epoch 1/50, Loss: 0.2301, Accuracy: 0.1003
Epoch 2/50, Loss: 0.2257, Accuracy: 0.1988
Epoch 3/50, Loss: 0.2214, Accuracy: 0.3018
Epoch 4/50, Loss: 0.2173, Accuracy: 0.4317
Epoch 5/50, Loss: 0.2133, Accuracy: 0.5405
Epoch 6/50, Loss: 0.2093, Accuracy: 0.6126
Epoch 7/50, Loss: 0.2053, Accuracy: 0.6544
Epoch 8/50, Loss: 0.2012, Accuracy: 0.6809
Epoch 9/50, Loss: 0.1972, Accuracy: 0.6995
Epoch 10/50, Loss: 0.1931, Accuracy: 0.7136
Epoch 11/50, Loss: 0.1889, Accuracy: 0.7253
Epoch 12/50, Loss: 0.1847, Accuracy: 0.7337
Epoch 13/50, Loss: 0.1804, Accuracy: 0.7406
Epoch 14/50, Loss: 0.1762, Accuracy: 0.7477
Epoch 15/50, Loss: 0.1719, Accuracy: 0.7534
Epoch 16/50, Loss: 0.1676, Accuracy: 0.7588
Epoch 17/50, Loss: 0.1634, Accuracy: 0.7635
Epoch 18/50, Loss: 0.1591, Accuracy: 0.7683
Epoch 19/50, Loss: 0.1550, Accuracy: 0.7722
Epoch 20/50, Loss: 0.1508, Accuracy: 0.7760
Epoch 21/50, Loss: 0.1468, Accuracy: 0.7798
Epoch 22/50, Loss: 0.1428, Accuracy: 0.7840
Epoch 23/50, Loss: 0.1390, Accuracy: 0.7870
Epoch 24/50, Loss: 0.1352, Accuracy: 0.7907
Epoch 25/50, Loss: 0.1316, Accuracy: 0.7941
Epoch 26/50, Loss: 0.1281, Accuracy: 0.7971
Epoch 27/50, Loss: 0.1248, Accuracy: 0.8003
Epoch 28/50, Loss: 0.1215, Accuracy: 0.8028
Epoch 29/50, Loss: 0.1184, Accuracy: 0.8053
Epoch 30/50, Loss: 0.1155, Accuracy: 0.8077
Epoch 31/50, Loss: 0.1127, Accuracy: 0.8103
Epoch 32/50, Loss: 0.1099, Accuracy: 0.8127
Epoch 33/50, Loss: 0.1074, Accuracy: 0.8149
Epoch 34/50, Loss: 0.1049, Accuracy: 0.8172
Epoch 35/50, Loss: 0.1026, Accuracy: 0.8193
Epoch 36/50, Loss: 0.1004, Accuracy: 0.8209
Epoch 37/50, Loss: 0.0982, Accuracy: 0.8230
Epoch 38/50, Loss: 0.0962, Accuracy: 0.8249
Epoch 39/50, Loss: 0.0943, Accuracy: 0.8270
Epoch 40/50, Loss: 0.0925, Accuracy: 0.8286
Epoch 41/50, Loss: 0.0907, Accuracy: 0.8302
Epoch 42/50, Loss: 0.0890, Accuracy: 0.8316
Epoch 43/50, Loss: 0.0875, Accuracy: 0.8333
Epoch 44/50, Loss: 0.0859, Accuracy: 0.8346
Epoch 45/50, Loss: 0.0845, Accuracy: 0.8358
Epoch 46/50, Loss: 0.0831, Accuracy: 0.8373
Epoch 47/50, Loss: 0.0818, Accuracy: 0.8383
Epoch 48/50, Loss: 0.0805, Accuracy: 0.8399
Epoch 49/50, Loss: 0.0793, Accuracy: 0.8411
Epoch 50/50, Loss: 0.0782, Accuracy: 0.8422
Training finished.
Model weights saved to model_weights.npz
```

### Тестирование модели (test.py)

```bash
python test.py --data_path mnist_data \
               --weights_path model_weights.npz
```

Параметры:

- `--data_path`: путь к директории с тестовыми данными
- `--weights_path`: путь к сохраненным весам модели

Для текущих параметров вывод будет выглядеть примерно так:

```
Number of test samples: 10000
Model weights loaded from model_weights.npz
Testing model...
Accuracy: 0.8541
Precision: 0.8552
Recall: 0.8507
F1 Score: 0.8503
```

### Предсказание на отдельном изображении (predict.py)

```bash
python predict.py --image_path path/to/image.png \
                  --weights_path model_weights.npz
```

Параметры:

- `--image_path`: путь к изображению для классификации
- `--weights_path`: путь к сохраненным весам модели

Для текущих параметров вывод будет выглядеть примерно так:

```
Model weights loaded from model_weights.npz
Prediction for image: path/to/image.png
Predicted class: 7
Confidence: 0.8465
```

## Требования к датасету

Датасет должен иметь следующую структуру:

```
dataset/
├── train/
│   ├── images/    # Изображения для обучения
│   └── labels/    # Метки для обучения
└── test/
    ├── images/    # Изображения для тестирования
    └── labels/    # Метки для тестирования
```

Требования к изображениям:

- Формат: одноканальные изображения (оттенки серого)

Требования к меткам:

- Формат: текстовый файл с метками для каждого изображения

### Генерация собственного датасета

Для генерации собственного датасета используйте скрипт generate.py:

```bash
python generate.py
```

Параметры:

- `--output_dir`: директория для сохранения сгенерированного датасета
- `--num_samples`: количество генерируемых образцов

## Метрики оценки

При тестировании модели вычисляются следующие метрики:

- Accuracy (точность)
- Precision (точность по классам)
- Recall (полнота)
- F1-score (F1-мера)
