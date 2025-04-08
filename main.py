# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from pathlib import Path

# Configuración de las rutas de las carpetas de entrada y salida
input_train_dir = 'dataset/train/'
input_test_dir = 'dataset/test/'
output_dir = 'processed_images/'

# Asegurarse de que las carpetas de salida existan
Path(os.path.join(output_dir, 'train', 'fractured')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_dir, 'train', 'not_fractured')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_dir, 'test', 'fractured')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_dir, 'test', 'not_fractured')).mkdir(parents=True, exist_ok=True)

# Función de preprocesamiento de una sola imagen
def preprocess_image(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)

    # Convertir a escala de grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ajuste de contraste (ecualización de histograma)
    image_eq = cv2.equalizeHist(image_gray)

    # Reducción de ruido con filtro gaussiano
    image_blur = cv2.GaussianBlur(image_eq, (5, 5), 0)

    # Detección de bordes con Canny
    edges = cv2.Canny(image_blur, 100, 200)

    # Normalización de la imagen (escala de 0 a 1)
    image_normalized = edges / 255.0  # Si es necesario para la red neuronal

    return image_normalized

# Función para procesar todas las imágenes en una carpeta
def process_images(input_dir, output_dir):
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)

        if os.path.isdir(class_path):  # Verificar que es una carpeta
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)

                if image_path.endswith('.jpg') or image_path.endswith('.png'):  # Solo procesar imágenes
                    print(f'Procesando {image_path}')
                    # Preprocesar la imagen
                    processed_image = preprocess_image(image_path)

                    # Guardar la imagen procesada en la carpeta de salida
                    output_image_path = os.path.join(output_dir, class_folder, image_name)
                    cv2.imwrite(output_image_path, (processed_image * 255).astype(np.uint8))  # Convertir a 0-255 para guardarla

# Preprocesar las imágenes de entrenamiento
process_images(input_train_dir, output_dir)

# Preprocesar las imágenes de prueba
process_images(input_test_dir, output_dir)

print("Preprocesamiento completo.")
