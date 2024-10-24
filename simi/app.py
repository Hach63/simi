import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# Configuration de l'application Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Charger le modèle pré-entraîné MobileNetV2 pour la classification
base_model = MobileNetV2(weights="imagenet", include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)

# Dictionnaire des catégories (exemple)
CATEGORIES = {
    'robe': 0,
    'chemise': 1,
    'pantalon': 2,
    # Ajouter d'autres catégories si nécessaire
}

# Fonction pour prédire la catégorie d'une image
def predict_category(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Taille standard pour MobileNetV2
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Extraire les caractéristiques de l'image
    features = model.predict(image)
    
    # Logique de classification pour la catégorie
    # Pour simplifier, nous utilisons ici un simple threshold, mais cela peut être remplacé par un modèle plus complexe
    # Ajouter ici la logique pour déterminer la catégorie en fonction des caractéristiques
    # Par exemple, retourner la catégorie prédite en utilisant un modèle entraîné sur vos catégories
    predicted_category = 'robe'  # Placeholder pour l'exemple
    return predicted_category

# Fonction pour lire une image et la convertir en vecteur
def image_to_vector(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Lire l'image en niveau de gris
    image = cv2.resize(image, (100, 100))  # Redimensionner à 100x100 pour la standardisation
    return image.flatten()  # Convertir l'image en un vecteur 1D

# Charger les images dans un dataset et créer des étiquettes avec leur catégorie
def load_images():
    image_data = {'robe': [], 'chemise': [], 'pantalon': []}
    image_labels = {'robe': [], 'chemise': [], 'pantalon': []}
    
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            path = os.path.join(UPLOAD_FOLDER, filename)
            category = predict_category(path)
            image_vector = image_to_vector(path)
            image_data[category].append(image_vector)
            image_labels[category].append(filename)

    # Crée des modèles KNN pour chaque catégorie
    knn_models = {}
    for category in CATEGORIES.keys():
        if len(image_data[category]) > 0:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(image_data[category], image_labels[category])
            knn_models[category] = knn
    
    return knn_models

# Charger les données initiales
knn_models = load_images()

# Route pour la page de téléchargement
@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Sauvegarder le fichier téléchargé
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Prédire la catégorie de l'image téléchargée
            category = predict_category(filepath)

            # Comparer l'image téléchargée avec les images existantes de la même catégorie
            uploaded_vector = image_to_vector(filepath)
            if category in knn_models:
                similar_images = knn_models[category].kneighbors([uploaded_vector], n_neighbors=3, return_distance=False)
                result = [knn_models[category].classes_[i] for i in similar_images[0]]
            else:
                result = []

            return render_template('upload.html', filename=file.filename, result=result, category=category)

    return render_template('upload.html')

# Lancer l'application
if __name__ == '__main__':
    app.run(debug=True)
