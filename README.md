# Système de Détection de Fatigue en Temps Réel

Ce système utilise la vision par ordinateur pour détecter les signes de fatigue chez un conducteur en analysant les mouvements des yeux et de la bouche en temps réel.

## Fonctionnalités

- Détection en temps réel des signes de fatigue
- Analyse du ratio d'aspect des yeux (EAR - Eye Aspect Ratio)
- Détection des bâillements
- Calibration automatique personnalisée
- Interface graphique moderne et intuitive
- Support des modes image et vidéo
- Adaptation à l'orientation du visage

## Technologies Utilisées

- Python 3.8+
- OpenCV (cv2) pour le traitement d'images
- dlib pour la détection faciale
- NumPy pour les calculs mathématiques
- CustomTkinter pour l'interface graphique
- PIL (Python Imaging Library) pour la manipulation d'images

## Installation

1. Cloner le repository :
```powershell
git clone https://github.com/votre-username/detection-somnolence.git
cd detection-somnolence
```

2. Installer les dépendances :
```powershell
pip install -r requirements.txt
```

3. Télécharger le fichier de points de repère faciaux :
- Télécharger `shape_predictor_68_face_landmarks.dat`
- Le placer dans le répertoire racine du projet

## Utilisation

1. Lancer l'application :
```powershell
python app_last_version.py
```

2. Fonctionnalités disponibles :
- **Load Image** : Charger une image depuis le disque
- **Process Image** : Analyser l'image chargée
- **Start Video** : Démarrer l'analyse en temps réel via webcam
- **Reset Calibration** : Réinitialiser la calibration du système

## Configuration

Les paramètres de détection peuvent être ajustés dans le dictionnaire CONFIG :

```python
CONFIG = {
    "EAR_THRESHOLD": 0.25,         # Seuil de fermeture des yeux
    "LIP_DISTANCE_THRESHOLD": 25,   # Seuil de bâillement
    "EYE_CLOSED_CONSECUTIVE_FRAMES": 8,   # Frames consécutives
    "MOUTH_OPEN_CONSECUTIVE_FRAMES": 10,  # Frames consécutives
    "EAR_CALIBRATION_FRAMES": 20,   # Frames de calibration
    "LIP_CALIBRATION_FRAMES": 20    # Frames de calibration
}
```

## Structure du Projet

```
detection-somnolence/
│
├── app_last_version.py           # Application principale
├── shape_predictor_68_face_landmarks.dat  # Modèle de points de repère
├── README.md                     # Documentation
├── requirements.txt              # Dépendances
└── img_test/                     # Images de test
    ├── face1.jpeg
    ├── face2.jpeg
    └── face3.jpg
```

## Caractéristiques Techniques

- Détection faciale avec dlib
- Calcul du Eye Aspect Ratio (EAR)
- Mesure de la distance labiale
- Calibration adaptative
- Compensation de l'orientation du visage
- Interface graphique moderne avec CustomTkinter

## Auteur

Mouhamed Diop

## Licence

Ce projet est sous licence MIT

---

*Développé dans le cadre du cours de Computer Vision à l'EPT*
