import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import dlib
import time
from collections import deque

# Configuration de CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Charger le détecteur de visage et le prédicteur de points clés
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Variables de configuration pour la détection de fatigue
CONFIG = {
    "EAR_THRESHOLD": 0.25,         # Seuil plus élevé pour être plus sensible
    "LIP_DISTANCE_THRESHOLD": 25,  # Ajusté pour être plus réaliste
    "EYE_CLOSED_CONSECUTIVE_FRAMES": 8,  # Pour la détection temporelle
    "MOUTH_OPEN_CONSECUTIVE_FRAMES": 10,  # Pour la détection temporelle
    "EAR_CALIBRATION_FRAMES": 20,  # Pour la calibration individuelle
    "LIP_CALIBRATION_FRAMES": 20,  # Pour la calibration individuelle
}

# Variables globales pour suivre l'état temporel
eye_closed_counter = 0
mouth_open_counter = 0
ear_history = deque(maxlen=CONFIG["EAR_CALIBRATION_FRAMES"])
lip_history = deque(maxlen=CONFIG["LIP_CALIBRATION_FRAMES"])
calibration_done = False
baseline_ear = None
baseline_lip = None

# Fonctions d'analyse améliorées
def eye_aspect_ratio(eye):
    # Calcul amélioré du EAR avec pondération
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    
    # Éviter la division par zéro
    if C < 0.1:
        return 0.0
    
    # Formule améliorée avec pondération
    ear = (A + B) / (2.0 * C)
    return ear

def lip_distance(mouth):
    # Calcul amélioré avec plusieurs points pour plus de robustesse
    top_lip = np.mean([mouth[13], mouth[14]], axis=0)
    bottom_lip = np.mean([mouth[19], mouth[18]], axis=0)
    distance = np.linalg.norm(top_lip - bottom_lip)
    
    # Normaliser par rapport à la largeur de la bouche pour tenir compte
    # des différences de taille du visage et de distance à la caméra
    mouth_width = np.linalg.norm(mouth[0] - mouth[6])
    if mouth_width < 0.1:
        return 0.0
    
    return distance / mouth_width * 100

def resize_image(image, max_size=(600, 600)):
    h, w = image.shape[:2]
    ratio = min(max_size[0] / w, max_size[1] / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

def detect_faces_dlib(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # utiliser différentes échelles pour une meilleure détection
    faces = face_detector(gray, 0)
    if len(faces) == 0:
        # Essayer avec une échelle différente si aucun visage n'est détecté
        faces = face_detector(gray, 1)
    return faces

def calibrate_metrics(ear_value, lip_dist_value):
    """Calibrer les métriques en fonction de l'utilisateur actuel"""
    global ear_history, lip_history, calibration_done, baseline_ear, baseline_lip
    
    if ear_value is not None:
        ear_history.append(ear_value)
    
    if lip_dist_value is not None:
        lip_history.append(lip_dist_value)
    
    # Quand suffisamment d'échantillons sont collectés, calculer les bases de référence
    if (len(ear_history) >= CONFIG["EAR_CALIBRATION_FRAMES"] and 
        len(lip_history) >= CONFIG["LIP_CALIBRATION_FRAMES"] and 
        not calibration_done):
        
        # Utiliser la médiane pour minimiser l'effet des valeurs aberrantes
        baseline_ear = np.median(ear_history)
        baseline_lip = np.median(lip_history)
        
        # Ajuster les seuils en fonction des bases de référence
        CONFIG["EAR_THRESHOLD"] = baseline_ear * 0.75  # 75% de la valeur normale
        CONFIG["LIP_DISTANCE_THRESHOLD"] = baseline_lip * 1.5  # 150% de la valeur normale
        
        calibration_done = True
        print(f"Calibration complete: EAR threshold = {CONFIG['EAR_THRESHOLD']:.3f}, LIP threshold = {CONFIG['LIP_DISTANCE_THRESHOLD']:.3f}")

def get_face_orientation(landmarks):
    """Déterminer l'orientation du visage pour ajuster les seuils"""
    # Points de repère pour le calcul de l'orientation (nez, menton, tempes)
    nose_tip = landmarks[33]
    chin = landmarks[8] 
    left_temple = landmarks[0]
    right_temple = landmarks[16]
    
    # Calculer l'angle de rotation horizontal (yaw)
    face_width = np.linalg.norm(right_temple - left_temple)
    nose_deviation = (nose_tip[0] - (left_temple[0] + right_temple[0])/2) / (face_width/2)
    
    # Retourne un facteur d'ajustement basé sur l'orientation
    # Plus le visage est tourné, plus le facteur est élevé
    adjustment_factor = 1.0 + abs(nose_deviation) * 0.3
    
    return adjustment_factor, nose_deviation

# def detect_fatigue(image, is_video=False):
#     global eye_closed_counter, mouth_open_counter
    
#     faces = detect_faces_dlib(image)
    
#     if len(faces) == 0:
#         return image, "No Face Detected", None, None

#     fatigue_level = 0  # 0: none, 1: mild, 2: moderate, 3: severe
#     fatigue_result = "No Fatigue Detected"
#     ear_value = None
#     lip_dist_value = None
#     fatigue_details = []

#     for face in faces:
#         landmarks = predictor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), face)
#         landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

#         left_eye = landmarks[42:48]
#         right_eye = landmarks[36:42]
#         mouth = landmarks[48:68]

#         # Calculer l'orientation du visage et obtenir le facteur d'ajustement
#         orientation_factor, yaw = get_face_orientation(landmarks)
        
#         # Ajuster les seuils en fonction de l'orientation
#         ear_threshold = CONFIG["EAR_THRESHOLD"] / orientation_factor
#         lip_threshold = CONFIG["LIP_DISTANCE_THRESHOLD"] * orientation_factor

#         # Calculer EAR et distance des lèvres
#         left_ear = eye_aspect_ratio(left_eye)
#         right_ear = eye_aspect_ratio(right_eye)
        
#         # Si le visage est tourné, pondérer en faveur de l'œil le plus visible
#         if yaw > 0.1:  # Visage tourné vers la droite
#             ear_value = left_ear * 0.7 + right_ear * 0.3
#         elif yaw < -0.1:  # Visage tourné vers la gauche
#             ear_value = left_ear * 0.3 + right_ear * 0.7
#         else:
#             ear_value = (left_ear + right_ear) / 2.0

#         lip_dist_value = lip_distance(mouth)
        
#         # Calibrer si nécessaire
#         if not calibration_done or is_video:
#             calibrate_metrics(ear_value, lip_dist_value)

#         # Dessiner le contour du visage
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#         # Dessiner les points des yeux et de la bouche
#         for i in range(36, 48):  # Yeux
#             cv2.circle(image, (landmarks[i][0], landmarks[i][1]), 2, (0, 255, 255), -1)
#         for i in range(48, 68):  # Bouche
#             cv2.circle(image, (landmarks[i][0], landmarks[i][1]), 2, (0, 165, 255), -1)

#         # Vérifier la fermeture des yeux
#         if ear_value < ear_threshold:
#             if is_video:
#                 eye_closed_counter += 1
#             if eye_closed_counter >= CONFIG["EYE_CLOSED_CONSECUTIVE_FRAMES"] or not is_video:
#                 cv2.putText(image, "Eyes Closed", (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 fatigue_level = max(fatigue_level, 2)
#                 fatigue_details.append("Eyes Closed")
#         else:
#             eye_closed_counter = max(0, eye_closed_counter - 1)

#         # Vérifier le bâillement
#         if lip_dist_value > lip_threshold:
#             if is_video:
#                 mouth_open_counter += 1
#             if mouth_open_counter >= CONFIG["MOUTH_OPEN_CONSECUTIVE_FRAMES"] or not is_video:
#                 cv2.putText(image, "Yawning", (x, y - 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 fatigue_level = max(fatigue_level, 2)
#                 fatigue_details.append("Yawning")
#         else:
#             mouth_open_counter = max(0, mouth_open_counter - 1)
            
#         # Afficher les métriques sur l'image
#         cv2.putText(image, f"EAR: {ear_value:.2f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(image, f"Lip: {lip_dist_value:.2f}", (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # Si calibration effectuée, afficher les seuils
#         if calibration_done:
#             cv2.putText(image, f"EAR Thresh: {ear_threshold:.2f}", (10, 90),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             cv2.putText(image, f"Lip Thresh: {lip_threshold:.2f}", (10, 120),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     # Déterminer le résultat final
#     if fatigue_level == 0:
#         fatigue_result = "No Fatigue Detected"
#     elif fatigue_level == 1:
#         fatigue_result = "Mild Fatigue Detected"
#     elif fatigue_level == 2:
#         fatigue_result = f"Fatigue Detected: {', '.join(fatigue_details)}"
#     elif fatigue_level == 3:
#         fatigue_result = f"Severe Fatigue Detected: {', '.join(fatigue_details)}"

#     return image, fatigue_result, ear_value, lip_dist_value

def detect_fatigue(image, is_video=False):
    global eye_closed_counter, mouth_open_counter
    
    faces = detect_faces_dlib(image)
    
    if len(faces) == 0:
        return image, "No Face Detected", None, None

    fatigue_level = 0  # 0: none, 1: mild, 2: moderate, 3: severe
    fatigue_result = "No Fatigue Detected"
    ear_value = None
    lip_dist_value = None
    fatigue_details = []

    for face in faces:
        landmarks = predictor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]
        mouth = landmarks[48:68]

        # Calculer l'orientation du visage et obtenir le facteur d'ajustement
        orientation_factor, yaw = get_face_orientation(landmarks)
        
        # Ajuster les seuils en fonction de l'orientation
        ear_threshold = CONFIG["EAR_THRESHOLD"] / orientation_factor
        lip_threshold = CONFIG["LIP_DISTANCE_THRESHOLD"] * orientation_factor

        # Calculer EAR et distance des lèvres
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Si le visage est tourné, pondérer en faveur de l'œil le plus visible
        if yaw > 0.1:  # Visage tourné vers la droite
            ear_value = left_ear * 0.7 + right_ear * 0.3
        elif yaw < -0.1:  # Visage tourné vers la gauche
            ear_value = left_ear * 0.3 + right_ear * 0.7
        else:
            ear_value = (left_ear + right_ear) / 2.0

        lip_dist_value = lip_distance(mouth)
        
        # Calibrer si nécessaire
        if not calibration_done or is_video:
            calibrate_metrics(ear_value, lip_dist_value)

        # Dessiner le contour du visage
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Dessiner les points des yeux et de la bouche
        for i in range(36, 48):  # Yeux
            cv2.circle(image, (landmarks[i][0], landmarks[i][1]), 2, (0, 255, 255), -1)
        for i in range(48, 68):  # Bouche
            cv2.circle(image, (landmarks[i][0], landmarks[i][1]), 2, (0, 165, 255), -1)

        # Vérifier la fermeture des yeux
        if ear_value <= 0.25:  # EAR inférieur ou égal à 0.25
            if is_video:
                eye_closed_counter += 1
            if eye_closed_counter >= CONFIG["EYE_CLOSED_CONSECUTIVE_FRAMES"] or not is_video:
                cv2.putText(image, "Eyes Closed", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                fatigue_level = max(fatigue_level, 2)
                fatigue_details.append("Eyes Closed")
        else:
            eye_closed_counter = max(0, eye_closed_counter - 1)

        # Vérifier le bâillement
        if lip_dist_value > lip_threshold:
            if is_video:
                mouth_open_counter += 1
            if mouth_open_counter >= CONFIG["MOUTH_OPEN_CONSECUTIVE_FRAMES"] or not is_video:
                cv2.putText(image, "Yawning", (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                fatigue_level = max(fatigue_level, 2)
                fatigue_details.append("Yawning")
        else:
            mouth_open_counter = max(0, mouth_open_counter - 1)
            
        # Afficher les métriques sur l'image
        cv2.putText(image, f"EAR: {ear_value:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Lip: {lip_dist_value:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Si calibration effectuée, afficher les seuils
        if calibration_done:
            cv2.putText(image, f"EAR Thresh: {ear_threshold:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f"Lip Thresh: {lip_threshold:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Déterminer le résultat final
    if fatigue_level == 0:
        fatigue_result = "No Fatigue Detected"
    elif fatigue_level == 1:
        fatigue_result = "Mild Fatigue Detected"
    elif fatigue_level == 2:
        fatigue_result = f"Fatigue Detected: {', '.join(fatigue_details)}"
    elif fatigue_level == 3:
        fatigue_result = f"Severe Fatigue Detected: {', '.join(fatigue_details)}"

    return image, fatigue_result, ear_value, lip_dist_value

# Interface graphique améliorée
class FatigueDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Fatigue Detection System")
        self.root.geometry("1100x800")
        self.root.minsize(900, 700)
        
        self.image = None
        self.video_capture = None
        self.is_video_mode = False
        self.video_running = False
        self.setup_ui()
    
    def setup_ui(self):
        # Créer une barre de titre personnalisée
        title_bar = ctk.CTkFrame(self.root, height=60, fg_color="#2D3250", corner_radius=0)
        title_bar.pack(fill="x", padx=0, pady=0)

        title_label = ctk.CTkLabel(
            title_bar, 
            text="ADVANCED FATIGUE DETECTION SYSTEM", 
            font=("Helvetica", 22, "bold"), 
            text_color="white"
        )
        title_label.pack(pady=12)

        # Créer un conteneur principal avec deux colonnes
        main_container = ctk.CTkFrame(self.root, fg_color="#111111")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Colonne de gauche (image)
        left_column = ctk.CTkFrame(main_container, fg_color="#F7F7F7", corner_radius=15)
        left_column.pack(side="left", fill="both", expand=True, padx=(5, 5), pady=5)

        # Cadre pour l'image
        image_frame = ctk.CTkFrame(left_column, fg_color="#EFEFEF", corner_radius=12)
        image_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Label pour afficher l'image
        self.image_label = ctk.CTkLabel(
            image_frame, 
            text="No image loaded", 
            font=("Helvetica", 16),
            corner_radius=8
        )
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Cadre pour les boutons sous l'image
        button_frame = ctk.CTkFrame(left_column, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 20))

        # Boutons améliorés
        self.load_button = ctk.CTkButton(
            button_frame, 
            text="Load Image",
            command=self.load_image,
            font=("Helvetica", 14, "bold"),
            fg_color="#3F4E75",
            hover_color="#5A699F",
            text_color="white",
            corner_radius=8,
            height=40,
            width=120
        )
        self.load_button.pack(side="left", padx=(0, 10))

        self.process_button = ctk.CTkButton(
            button_frame, 
            text="Process Image",
            command=self.process_image,
            font=("Helvetica", 14, "bold"),
            fg_color="#3F4E75",
            hover_color="#5A699F",
            text_color="white",
            corner_radius=8,
            height=40,
            width=120,
            state="disabled"
        )
        self.process_button.pack(side="left", padx=(0, 10))
        
        self.video_button = ctk.CTkButton(
            button_frame, 
            text="Start Video",
            command=self.toggle_video,
            font=("Helvetica", 14, "bold"),
            fg_color="#3F4E75",
            hover_color="#5A699F",
            text_color="white",
            corner_radius=8,
            height=40,
            width=120
        )
        self.video_button.pack(side="left", padx=(0, 10))
        
        self.calibrate_button = ctk.CTkButton(
            button_frame, 
            text="Reset Calibration",
            command=self.reset_calibration,
            font=("Helvetica", 14, "bold"),
            fg_color="#3F4E75",
            hover_color="#5A699F",
            text_color="white",
            corner_radius=8,
            height=40,
            width=120
        )
        self.calibrate_button.pack(side="left")

        # Colonne de droite (résultats)
        right_column = ctk.CTkFrame(main_container, fg_color="transparent", width=800)
        right_column.pack(side="right", fill="both", padx=(5, 5), pady=5)

        # Cadre des résultats
        self.result_frame = ctk.CTkFrame(right_column, fg_color="#D0E8FF", corner_radius=15)
        self.result_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # Titre du cadre de résultats
        result_title = ctk.CTkLabel(
            self.result_frame, 
            text="Detection Results", 
            font=("Helvetica", 18, "bold"),
            text_color="#424874"
        )
        result_title.pack(pady=(20, 15))

        # Label pour afficher le résultat de la détection de fatigue
        self.fatigue_label = ctk.CTkLabel(
            self.result_frame, 
            text="No analysis yet", 
            font=("Helvetica", 16, "bold"),
            corner_radius=8,
            fg_color="#E8E8E8",
            text_color="#424874",
            height=40
        )
        self.fatigue_label.pack(pady=(5, 20), padx=25, fill="x")

        # Créer un cadre pour les métriques
        metrics_frame = ctk.CTkFrame(self.result_frame, fg_color="transparent")
        metrics_frame.pack(fill="x", expand=True, padx=25, pady=(0, 20))

        # Sous-cadre pour EAR
        ear_frame = ctk.CTkFrame(metrics_frame, fg_color="#E8E8E8", corner_radius=8)
        ear_frame.pack(fill="x", pady=(0, 10))

        self.ear_label = ctk.CTkLabel(
            ear_frame, 
            text="EAR: -", 
            font=("Helvetica", 14),
            text_color="#424874"
        )
        self.ear_label.pack(pady=(10, 5), padx=15, anchor="w")

        self.ear_progress = ctk.CTkProgressBar(ear_frame, height=15, corner_radius=5, progress_color="#1E88E5")
        self.ear_progress.pack(pady=(0, 10), padx=15, fill="x")
        self.ear_progress.set(0)  # Valeur initiale
        
        # Label pour le seuil EAR
        self.ear_threshold_label = ctk.CTkLabel(
            ear_frame, 
            text="Threshold: -", 
            font=("Helvetica", 12),
            text_color="#424874"
        )
        self.ear_threshold_label.pack(pady=(0, 10), padx=15, anchor="w")

        # Sous-cadre pour Lip Distance
        lip_frame = ctk.CTkFrame(metrics_frame, fg_color="#E8E8E8", corner_radius=8)
        lip_frame.pack(fill="x")

        self.lip_distance_label = ctk.CTkLabel(
            lip_frame, 
            text="Lip Distance: -", 
            font=("Helvetica", 14),
            text_color="#424874"
        )
        self.lip_distance_label.pack(pady=(10, 5), padx=15, anchor="w")

        self.lip_progress = ctk.CTkProgressBar(lip_frame, height=15, corner_radius=5, progress_color="#1E88E5")
        self.lip_progress.pack(pady=(0, 10), padx=15, fill="x")
        self.lip_progress.set(0)  # Valeur initiale
        
        # Label pour le seuil lip distance
        self.lip_threshold_label = ctk.CTkLabel(
            lip_frame, 
            text="Threshold: -", 
            font=("Helvetica", 12),
            text_color="#424874"
        )
        self.lip_threshold_label.pack(pady=(0, 10), padx=15, anchor="w")

        # Informations supplémentaires
        info_frame = ctk.CTkFrame(self.result_frame, fg_color="#E8E8E8", corner_radius=8)
        info_frame.pack(fill="x", padx=25, pady=(0, 20))

        info_title = ctk.CTkLabel(
            info_frame, 
            text="Fatigue Detection Information", 
            font=("Helvetica", 14, "bold"),
            text_color="#424874"
        )
        info_title.pack(pady=(10, 5))

        info_text = ctk.CTkLabel(
            info_frame, 
            text="• EAR: Eye Aspect Ratio (lower = more closed)\n• Calibration: System adapts to your facial features\n• Face Orientation: Detection adjusts to head position\n• Video Mode: Tracks blinking and yawning over time", 
            font=("Helvetica", 12),
            text_color="#424874",
            justify="left"
        )
        info_text.pack(pady=(0, 10), padx=15, anchor="w")
        
        # Ajouter une section pour l'état de calibration
        self.calibration_frame = ctk.CTkFrame(self.result_frame, fg_color="#E8E8E8", corner_radius=8)
        self.calibration_frame.pack(fill="x", padx=25, pady=(0, 20))
        
        self.calibration_label = ctk.CTkLabel(
            self.calibration_frame, 
            text="Calibration: Not Started", 
            font=("Helvetica", 14, "bold"),
            text_color="#424874"
        )
        self.calibration_label.pack(pady=10, padx=15)

        # Footer
        footer = ctk.CTkFrame(self.root, height=40, fg_color="#2D3250", corner_radius=0)
        footer.pack(fill="x", padx=0, pady=0)

        footer_label = ctk.CTkLabel(
            footer, 
            text="© 2025 Advanced Fatigue Detection System", 
            font=("Helvetica", 12),
            text_color="white"
        )
        footer_label.pack(pady=10)
    
    def reset_calibration(self):
        """Réinitialiser la calibration"""
        global ear_history, lip_history, calibration_done, baseline_ear, baseline_lip
        ear_history.clear()
        lip_history.clear()
        calibration_done = False
        baseline_ear = None
        baseline_lip = None
        
        # Réinitialiser les valeurs de configuration par défaut
        CONFIG["EAR_THRESHOLD"] = 0.25
        CONFIG["LIP_DISTANCE_THRESHOLD"] = 25
        
        self.update_calibration_status()
        messagebox.showinfo("Calibration", "Calibration has been reset.")
    
    def update_calibration_status(self):
        """Mettre à jour l'affichage de l'état de calibration"""
        if not calibration_done:
            ear_progress = len(ear_history) / CONFIG["EAR_CALIBRATION_FRAMES"] * 100
            lip_progress = len(lip_history) / CONFIG["LIP_CALIBRATION_FRAMES"] * 100
            avg_progress = (ear_progress + lip_progress) / 2
            
            self.calibration_label.configure(
                text=f"Calibration: In Progress ({avg_progress:.0f}%)",
                text_color="#E67E22"
            )
            
            # Mettre à jour les labels de seuil
            self.ear_threshold_label.configure(text=f"Threshold: {CONFIG['EAR_THRESHOLD']:.2f} (default)")
            self.lip_threshold_label.configure(text=f"Threshold: {CONFIG['LIP_DISTANCE_THRESHOLD']:.2f} (default)")
        else:
            self.calibration_label.configure(
                text=f"Calibration: Complete",
                text_color="#27AE60"
            )
            
            # Mettre à jour les labels de seuil
            self.ear_threshold_label.configure(text=f"Threshold: {CONFIG['EAR_THRESHOLD']:.2f} (calibrated)")
            self.lip_threshold_label.configure(text=f"Threshold: {CONFIG['LIP_DISTANCE_THRESHOLD']:.2f} (calibrated)")
    
    def load_image(self):
        """Charger une image depuis le disque"""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                messagebox.showerror("Error", "Failed to load image. Check the file path and integrity.")
                return
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image = resize_image(self.image, max_size=(600, 600))
            self.display_image(self.image)
            self.process_button.configure(state="normal")
            
            # Désactiver le mode vidéo s'il est actif
            if self.is_video_mode:
                self.toggle_video()
    
    def display_image(self, img, processed=False):
        """Afficher une image dans l'interface"""
        image_pil = Image.fromarray(img)
        ctk_image = ctk.CTkImage(light_image=image_pil, size=(image_pil.width, image_pil.height))
        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image
        
        if processed:
            self.update_results(img)
    
    def process_image(self):
        """Traiter l'image chargée pour la détection de fatigue"""
        if self.image is not None:
            processed_image = self.image.copy()
            processed_image, fatigue_result, ear_value, lip_dist_value = detect_fatigue(processed_image)
            self.display_image(processed_image, processed=True)
            self.update_results_data(fatigue_result, ear_value, lip_dist_value)
            self.update_calibration_status()
    
    def toggle_video(self):
        """Basculer entre le mode image et le mode vidéo"""
        if self.is_video_mode:
            # Arrêter le mode vidéo
            self.video_running = False
            self.is_video_mode = False
            self.video_button.configure(text="Start Video")
            
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
        else:
            # Démarrer le mode vidéo
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Error", "Could not open webcam. Check your camera connection.")
                return
            
            self.is_video_mode = True
            self.video_running = True
            self.video_button.configure(text="Stop Video")
            self.process_button.configure(state="disabled")
            
            # Lancer la boucle de capture vidéo
            self.process_video_frame()
    
    def process_video_frame(self):
        """Traiter une image de la webcam et programmer la prochaine"""
        if self.video_running and self.video_capture is not None:
            ret, frame = self.video_capture.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = resize_image(frame, max_size=(600, 600))
                
                processed_frame, fatigue_result, ear_value, lip_dist_value = detect_fatigue(frame, is_video=True)
                self.display_image(processed_frame, processed=False)
                self.update_results_data(fatigue_result, ear_value, lip_dist_value)
                self.update_calibration_status()
                
                # Programmer le prochain traitement
                self.root.after(30, self.process_video_frame)
    
    def update_results_data(self, fatigue_result, ear_value, lip_dist_value):
        """Mettre à jour les données de résultat dans l'interface"""
        self.fatigue_label.configure(text=f"{fatigue_result}")
        
        if ear_value is not None:
            self.ear_label.configure(text=f"EAR: {ear_value:.2f}")
            # Normaliser EAR: une valeur normale est environ 0.2-0.3
            # Plus la valeur est basse, plus les yeux sont fermés
            normalized_ear = min(max(ear_value / 0.4, 0), 1)
            self.ear_progress.set(normalized_ear)
            
        if lip_dist_value is not None:
            self.lip_distance_label.configure(text=f"Lip Distance: {lip_dist_value:.2f}")
            # Normaliser lip distance: plus la valeur est élevée

            # plus la bouche est ouverte
            normalized_lip = min(lip_dist_value / 50, 1)
            self.lip_progress.set(normalized_lip)
    
    def update_results(self, img):
        """Mettre à jour les résultats après le traitement de l'image"""
        processed_image, fatigue_result, ear_value, lip_dist_value = detect_fatigue(img)
        self.update_results_data(fatigue_result, ear_value, lip_dist_value)
        
        # Changer la couleur du panneau de résultats selon la détection
        if "Fatigue Detected" in fatigue_result:
            self.result_frame.configure(fg_color="#FFD1D1")  # Rouge clair pour la fatigue
            self.fatigue_label.configure(text_color="#D62828")
        elif "Mild Fatigue" in fatigue_result:
            self.result_frame.configure(fg_color="#FFECD6")  # Orange clair
            self.fatigue_label.configure(text_color="#E67E22")
        else:
            self.result_frame.configure(fg_color="#FFF0F0")  # Rose pâle
            self.fatigue_label.configure(text_color="#1B7A0C")

# Fonction principale pour lancer l'application
def main():
    root = ctk.CTk()
    app = FatigueDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()