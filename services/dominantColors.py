import cv2
import numpy as np
from sklearn.cluster import KMeans


def find_dominant_colors(img, nbreDominantColors=10):
    # Calculer et afficher les couleurs dominantes d’une image à couleurs réelles ( Calcul dans l’espace RGB)
    # Créer une image temopraire
    barColorW = 75
    barColorH = 50
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Changement d'echelle, pour avoir moins d'exemples
    width = 50  # largeur cible
    ratio = img.shape[0] / img.shape[1]
    height = int(img.shape[1] * ratio)
    dim = (width, height)
    img = cv2.resize(img, dim)
    # Paramètres d'apprentissage
    # Un triplet (B, G, R) par ligne
    examples = img.reshape((img.shape[0] * img.shape[1], 3))
    # Groupement par la technique des KMEANS
    kmeans = KMeans(n_clusters=nbreDominantColors, n_init=10)
    kmeans.fit(examples)
    # Les Centres des groupement représentent les couleurs dominantes (B, G, R)
    colors = kmeans.cluster_centers_.astype(int)
    dominant_colors = []
    for i in range(0, nbreDominantColors):
        dominant_colors.append([int(x) for x in colors[i]])
    return dominant_colors
