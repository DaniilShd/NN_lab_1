# Импортирую необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import skimage as io
import os
import tqdm
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
from tqdm.auto import tqdm
import tqdm
import random
import tabulate
from torchvision.datasets import MNIST

# MNIST label names
MNIST_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def get_MNIST(data_path):
    dataset = MNIST(root=f'{data_path}/', train=True, download=True)
    # Создание директории для сохранения изображений
    os.makedirs(f'{data_path}/data_MNIST/MNIST_images', exist_ok=True)

    # Сохранение изображений и создание CSV файла
    data = []
    for idx, (image, label) in enumerate(tqdm.tqdm(dataset)):
        # Сохранение изображения в формате PNG
        image_path = f'{data_path}/data_MNIST/MNIST_images/{idx}.png'
        image.save(image_path)
        data.append([image_path, label])

    # Сохранение данных в CSV файл
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    df.to_csv(f'{data_path}/data_MNIST/MNIST.csv', index=False)

class CustomDataset(Dataset):
  def __init__(self, csv_file, transform=None):
    self.data = pd.read_csv(csv_file)
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    sample = self.data.iloc[idx]
    image_path = sample['image_path']
    label = sample['label']

    image = Image.open(image_path).convert('RGB')

    if self.transform:
      image = self.transform(image)

    return image, label


def create_dataloader(dataset, batch_size=32, num_workers=2, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )

# Итерация по DataLoader
def iterate_dataloader(dataloader, device='cpu'):
    batch_stats = []
    start_time = time.time()

    for batch_idx, (data, targets) in enumerate(tqdm.tqdm(dataloader, desc='Processing')):
        # Перемещаем данные на устройство
        data, targets = data.to(device), targets.to(device)

        # Заглушка forward pass
        outputs = torch.randn(data.size(0), 10, device=device)  # Случайные выходы

        # Расчет лосса
        loss = dummy_loss_function(outputs, targets)

        # Логирование статистики
        batch_stats.append({
            'batch_idx': batch_idx,
            'loss': loss.item(),
            'time': time.time() - start_time
        })

    return batch_stats

# Оценка производительности пайплайна
def evaluate_pipeline_performance():
    batch_sizes = [1, 4, 16, 32]
    num_workers_list = [0, 2]
    results = []

    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            print(f"\nTesting batch_size={batch_size}, num_workers={num_workers}")

            dataloader = create_dataloader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )

            start_time = time.time()
            stats = iterate_dataloader(dataloader)
            elapsed_time = time.time() - start_time

            results.append({
                'batch_size': batch_size,
                'num_workers': num_workers,
                'total_time': elapsed_time,
                'avg_batch_time': elapsed_time / len(stats)
            })

    return results


# Создание своей loss функции
def dummy_loss_function(outputs, targets):
    loss_fn = torch.nn.CrossEntropyLoss()  # Для классификации
    return loss_fn(outputs, targets.long())  # targets должны быть long()

if __name__ == '__main__':
    base_path = './data'

    get_MNIST(base_path)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    CLASSES = [
        "healthy", "'sick_early'", "'sick_late'"
    ]

    dataset = CustomDataset(csv_file=f'{base_path}/data_MNIST/MNIST.csv',
                            transform=transform)

    dataloader = create_dataloader(dataset)

    result = evaluate_pipeline_performance()

    # Создаем таблицу с результатами
    results_df = pd.DataFrame(result)
    print("\nPerformance Results:")
    print(results_df.to_markdown(index=False))