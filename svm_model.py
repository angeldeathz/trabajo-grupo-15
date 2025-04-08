import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import joblib

# Directorios de las imágenes procesadas
input_train_dir = 'processed_images/train/'
input_test_dir = 'processed_images/test/'

# Función para preprocesar las imágenes y extraer características HOG
def preprocess_image(image_path, return_hog=False):
    # Cargar la imagen
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None

    # Redimensionar la imagen si es necesario
    image_resized = cv2.resize(image, (128, 128))  # Redimensionamos a 128x128

    # Aplicar operaciones morfológicas, si es necesario
    kernel = np.ones((5, 5), np.uint8)
    image_morphed = cv2.morphologyEx(image_resized, cv2.MORPH_CLOSE, kernel)  # Cerrar áreas pequeñas

    # Extraer características HOG
    hog_descriptor = cv2.HOGDescriptor()
    hog_features = hog_descriptor.compute(image_morphed)

    if return_hog:
        return hog_features
    return image_morphed

# Función para cargar imágenes y extraer HOG
def load_images_and_labels(input_dir, limit=10):
    images = []
    labels = []
    count = 0

    # Recorrer las carpetas 'fractured' y 'not_fractured'
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)
        if os.path.isdir(class_path):  # Asegurarse de que es una carpeta
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)

                if image_path.endswith('.jpg') or image_path.endswith('.png'):  # Solo procesar imágenes
                    print(f'Procesando {image_path}')
                    # Preprocesar la imagen y extraer HOG
                    hog_features = preprocess_image(image_path, return_hog=True)
                    if hog_features is not None:
                        images.append(hog_features.flatten())  # HOG es un vector 1D
                        labels.append(class_folder)  # 'fractured' o 'not_fractured'
                        # count += 1
                        # if count >= limit:  # Detenerse si alcanzamos el límite
                        #     break
    
    return np.array(images), np.array(labels)

# Cargar imágenes y etiquetas de entrenamiento
print("Cargando imágenes y etiquetas de entrenamiento...")
X_train, y_train = load_images_and_labels(input_train_dir, limit=2000)

# Cargar imágenes y etiquetas de prueba
print("Cargando imágenes y etiquetas de prueba...")
X_test, y_test = load_images_and_labels(input_test_dir, limit=200)

# Crear el modelo SVM
print("Entrenando el modelo SVM...")
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluar el modelo en las imágenes de prueba
y_pred = svm_model.predict(X_test)
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

# Guardar el modelo entrenado
joblib.dump(svm_model, 'svm_model.pkl')
print("Modelo SVM guardado como 'svm_model.pkl'")
