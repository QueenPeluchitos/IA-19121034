import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tkinter import Tk, simpledialog

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video
cap = cv2.VideoCapture(0)

# Lista ampliada de índices de landmarks específicos
selected_points = [33, 133, 362, 263, 61, 291, 4, 199, 234, 454, 10, 152]  # Incluye cejas, mandíbula, etc.

# Cargar datos existentes del archivo Excel
try:
    data = pd.read_excel("distancias_caras.xlsx")
except FileNotFoundError:
    print("Archivo de datos no encontrado. Se creará uno nuevo.")

def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def reconocer_rostro(medidas_actuales):
    """Compara las medidas actuales con las almacenadas para identificar el rostro."""
    if data.empty:
        return "Datos insuficientes para reconocimiento."

    # Calcular la distancia euclidiana entre las medidas actuales y las almacenadas
    data["Diferencia"] = data.apply(
        lambda row: np.linalg.norm([
            row["Distancia_Ojos1"] - medidas_actuales["Distancia_Ojos1"],
            row["Distancia_Ojos2"] - medidas_actuales["Distancia_Ojos2"],
            row["Distancia_Boca"] - medidas_actuales["Distancia_Boca"],
            row["Distancia_Nariz"] - medidas_actuales["Distancia_Nariz"],
            # Agregar más distancias aquí si se calculan
        ]),
        axis=1
    )

    # Identificar el nombre con la menor diferencia
    menor_diferencia = data.loc[data["Diferencia"].idxmin()]
    if menor_diferencia["Diferencia"] < 50:  # Umbral ajustable
        return menor_diferencia["Nombre"]
    return "Desconocido"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espejo para mayor naturalidad
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            puntos = {}
            
            for idx in selected_points:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                puntos[idx] = (x, y)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Dibuja el punto en verde
            
            # Calcular distancias adicionales
            if all(k in puntos for k in [33, 133, 362, 263, 61, 291, 4, 199, 234, 454, 10, 152]):
                d_ojos = distancia(puntos[33], puntos[133])
                d_ojos2 = distancia(puntos[362], puntos[263])
                d_boca = distancia(puntos[61], puntos[291])
                d_nariz = distancia(puntos[4], puntos[10])
                d_cejas = distancia(puntos[199], puntos[234])
                d_mandibula = distancia(puntos[10], puntos[152])

                medidas_actuales = {
                    "Distancia_Ojos1": d_ojos,
                    "Distancia_Ojos2": d_ojos2,
                    "Distancia_Boca": d_boca,
                    "Distancia_Nariz": d_nariz,
                    "Distancia_Cejas": d_cejas,
                    "Distancia_Mandibula": d_mandibula,
                }

                # Reconocer rostro
                nombre_reconocido = reconocer_rostro(medidas_actuales)
                cv2.putText(frame, f"Reconocido: {nombre_reconocido}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Mostrar distancias en pantalla
                cv2.putText(frame, f"D ojos: {int(d_ojos)}", (puntos[33][0], puntos[33][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"D ojos2: {int(d_ojos2)}", (puntos[362][0], puntos[362][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"D boca: {int(d_boca)}", (puntos[61][0], puntos[61][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"D nariz: {int(d_nariz)}", (puntos[4][0], puntos[4][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"D cejas: {int(d_cejas)}", (puntos[10][0], puntos[10][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"D mandibula: {int(d_mandibula)}", (puntos[199][0], puntos[199][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Guardar datos al presionar 's'
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    # Crear un cuadro de diálogo para ingresar el nombre
                    root = Tk()
                    root.withdraw()  
                    nombre = simpledialog.askstring("Nombre", "Ingrese el nombre del dueño de las distancias:")
                    root.destroy()

                    if nombre:
                        # Agregar datos al DataFrame
                        data = pd.concat([data, pd.DataFrame([{
                            "Nombre": nombre,
                            "Distancia_Ojos1": d_ojos,
                            "Distancia_Ojos2": d_ojos2,
                            "Distancia_Boca": d_boca,
                            "Distancia_Nariz": d_nariz,
                            "Distancia_Cejas": d_cejas,
                            "Distancia_Mandibula": d_mandibula
                        }])], ignore_index=True)

                        # Guardar en un archivo Excel
                        data.to_excel("distancias_caras.xlsx", index=False)
                        print(f"Datos guardados para {nombre}")

    cv2.imshow('PuntosFacialesMediaPipe', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()