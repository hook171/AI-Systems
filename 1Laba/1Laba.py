import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import os

# предварительная обработка тут
def preprocess_image(image_path, target_size):

    image = Image.open(image_path)
    # фильтр резкости
    image = image.filter(ImageFilter.SHARPEN)
    # изменение размера до target_size x target_size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return image

# создание маски изображения
def create_mask(image, target_size):

    # пример создания маски: выделение объекта по пороговому значению
    gray_image = image.convert('L')
    mask = gray_image.point(lambda p: 255 if p > 128 else 0)
    mask = mask.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return mask

# генератор аугментации изображений
def augment_image(image, mask):
    
    # случайный поворот
    angle = random.randint(-30, 30)
    image = image.rotate(angle)
    mask = mask.rotate(angle)

    # случайно обрезаем
    width, height = image.size
    crop_size = random.randint(int(0.8 * width), width)
    x = random.randint(0, width - crop_size)
    y = random.randint(0, height - crop_size)
    image = image.crop((x, y, x + crop_size, y + crop_size))
    mask = mask.crop((x, y, x + crop_size, y + crop_size))

    # случайное изменение цвета
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))

    # случайное размытие
    if random.random() > 0.5:
        image = image.filter(ImageFilter.BLUR)

    return image, mask

def create_dataset(image_path, target_size, num_augmented_images=100):
    # предварительная обработка изображения
    image = preprocess_image(image_path, target_size)
    mask = create_mask(image, target_size)

    if not os.path.exists('1Laba/augmented_images'):
        os.makedirs('1Laba/augmented_images')
    if not os.path.exists('1Laba/augmented_masks'):
        os.makedirs('1Laba/augmented_masks')

    # генерация аугментированных изображений и масок
    for i in range(num_augmented_images):
        augmented_image, augmented_mask = augment_image(image, mask)
        augmented_image.save(f'1Laba/augmented_images/augmented_{i}.png')
        augmented_mask.save(f'1Laba/augmented_masks/augmented_mask_{i}.png')

    print(f"Создано {num_augmented_images} аугментированных изображений и масок.")

n = 7
image_path = '1Laba/test.jpg'
target_size = 2**n  # степерь двойки для изображения
create_dataset(image_path, target_size)