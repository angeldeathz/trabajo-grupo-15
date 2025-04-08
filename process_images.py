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
def preprocess_image(image_path, return_hog=False):
    # Cargar la imagen
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ No se pudo cargar: {image_path}")
        return None

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ajuste de contraste (ecualización de histograma)
    equalized = cv2.equalizeHist(gray)

    # Reducción de ruido
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Detección de bordes
    edges = cv2.Canny(blurred, 100, 200)

    # Operación morfológica (closing para cerrar pequeños huecos)
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Normalización
    normalized = morphed / 255.0

    # Si queremos usar un modelo clásico como SVM, retornamos descriptores
    if return_hog:
        hog = cv2.HOGDescriptor()
        h = hog.compute(morphed)
        return h  # vector de características

    return normalized  # imagen procesada lista para red neuronal


# Función para procesar todas las imágenes en una carpeta
def process_images(input_dir, output_dir, max_images_per_class=1500):
    subdir = Path(input_dir).name  # "train" o "test"
    
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)

        if os.path.isdir(class_path):  # Verificar que es una carpeta
            processed_count = 0  # Contador para las imágenes procesadas

            for image_name in os.listdir(class_path):
                # Si ya se procesaron max_images_per_class imágenes, detenerse
                if processed_count >= max_images_per_class:
                    break
                
                image_path = os.path.join(class_path, image_name)

                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Más robusto
                    print(f'Procesando {image_path}')

                    # Preprocesar la imagen
                    processed_image = preprocess_image(image_path)

                    if processed_image is None:
                        print(f'❌ Imagen inválida: {image_path}')
                        continue

                    # Crear el path completo de salida
                    output_image_path = os.path.join(output_dir, subdir, class_folder, image_name)
                    output_dir_path = os.path.dirname(output_image_path)

                    if not os.path.exists(output_dir_path):
                        os.makedirs(output_dir_path, exist_ok=True)

                    success = cv2.imwrite(output_image_path, (processed_image * 255).astype(np.uint8))
                    if not success:
                        print(f'❌ No se pudo guardar la imagen en: {output_image_path}')

                    processed_count += 1  # Incrementar el contador después de procesar una imagen

# Preprocesar las imágenes de entrenamiento
process_images(input_train_dir, output_dir)

# Preprocesar las imágenes de prueba
process_images(input_test_dir, output_dir)

print("Preprocesamiento completo.")
