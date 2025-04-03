import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tkinter import Tk, simpledialog

# Inicializar MediaPipe Face Mesh y Face Detection
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Captura de video desde la webcam
cap = cv2.VideoCapture(0)

# Índices clave de landmarks faciales
selected_points = [33, 133, 362, 263, 61, 291, 4, 10, 13, 14, 17, 78, 308, 199, 234]

# Cargar datos existentes de reconocimiento facial
data = pd.read_excel("distancias_caras.xlsx") if pd.io.common.file_exists("distancias_caras.xlsx") else pd.DataFrame(columns=["Nombre", "Distancia_Ojos1", "Distancia_Ojos2", "Distancia_Boca", "Distancia_Nariz", "Distancia_Cejas"])

def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def reconocer_rostro(medidas_actuales):
    """Compara las medidas actuales con las almacenadas para identificar el rostro."""
    if data.empty:
        return "Datos insuficientes"
    data["Diferencia"] = data.apply(
        lambda row: np.linalg.norm([
            row["Distancia_Ojos1"] - medidas_actuales["Distancia_Ojos1"],
            row["Distancia_Ojos2"] - medidas_actuales["Distancia_Ojos2"],
            row["Distancia_Boca"] - medidas_actuales["Distancia_Boca"],
            row["Distancia_Nariz"] - medidas_actuales["Distancia_Nariz"],
            row["Distancia_Cejas"] - medidas_actuales["Distancia_Cejas"],
        ]),
        axis=1
    )
    menor_diferencia = data.loc[data["Diferencia"].idxmin()]
    return menor_diferencia["Nombre"] if menor_diferencia["Diferencia"] < 50 else "Desconocido"

def detectar_emocion(puntos):
    """Estimación de emociones basada en distancias faciales."""
    if all(k in puntos for k in [13, 14, 17, 78, 308, 199, 234]):
        d_cejas = distancia(puntos[199], puntos[234]) # Distancia entre cejas
        d_boca_abierta = distancia(puntos[14], puntos[17]) # Distancia entre boca abierta
        d_sonrisa = distancia(puntos[78], puntos[308])
        #Parametros de las emociones
        if d_boca_abierta > 25:
            return "Sorprendido"
        elif d_sonrisa < 50 and d_cejas < 10:
            return "Triste"
        elif d_cejas > 25 and d_boca_abierta < 10:
            return "Enojado"
        elif d_sonrisa > 50:
            return "Feliz"  
        else:
            return "Neutral"
    return "No detectado"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            puntos = {}
            for idx in selected_points:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                puntos[idx] = (x, y)
                cv2.circle(frame, (x, y), 2, (141, 216, 143), -1)
            
            if all(k in puntos for k in [33, 133, 362, 263, 61, 291, 4, 10, 199, 234]):
                medidas_actuales = {
                    "Distancia_Ojos1": distancia(puntos[33], puntos[133]),
                    "Distancia_Ojos2": distancia(puntos[362], puntos[263]),
                    "Distancia_Boca": distancia(puntos[61], puntos[291]),
                    "Distancia_Nariz": distancia(puntos[4], puntos[10]),
                    "Distancia_Cejas": distancia(puntos[199], puntos[234]),
                }
                nombre_reconocido = reconocer_rostro(medidas_actuales)
                emocion = detectar_emocion(puntos)
                
                cv2.putText(frame, f"Reconocido: {nombre_reconocido}", (10, 30), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (229, 145, 212), 2)
                cv2.putText(frame, f"Emocion: {emocion}", (10, 60), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 200, 0), 2)
                
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    root = Tk()
                    root.withdraw()
                    nombre = simpledialog.askstring("Nombre", "Dame el nombre:")
                    root.destroy()
                    if nombre:
                        data = pd.concat([data, pd.DataFrame([{**medidas_actuales, "Nombre": nombre}])], ignore_index=True)
                        data.to_excel("distancias_caras.xlsx", index=False)
                        print(f"Datos guardados para {nombre}")

    cv2.imshow('Reconocimiento Facial y Emociones', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()