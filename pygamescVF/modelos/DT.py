import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def entrenar_arbol_salto(datos_modelo):
    if len(datos_modelo) < 10:
        print("No hay suficientes datos para entrenar el árbol.")
        return None

    datos = np.array(datos_modelo)
    X = datos[:, :6] 
    y = datos[:, 6]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    arbol = DecisionTreeClassifier(max_depth=10)
    arbol.fit(X_train, y_train)

    accuracy = arbol.score(X_test, y_test)
    print(f"Precisión del árbol de salto: {accuracy:.2f}")

    return arbol

def decidir_salto_arbol(jugador, bala, velocidad_bala, bala_aire, bala_disparada_aire, arbol_salto, salto, en_suelo):
    if arbol_salto is None:
        print("Árbol no entrenado.")
        return salto, en_suelo

    distancia_suelo = abs(jugador.x - bala.x)
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)
    hay_bala_aire = 1 if bala_disparada_aire else 0

    entrada = np.array([[velocidad_bala, distancia_suelo, distancia_aire_x, distancia_aire_y, hay_bala_aire, jugador.x]])

    prediccion = arbol_salto.predict(entrada)[0]

    if prediccion == 1 and en_suelo:
        salto = True
        en_suelo = False

    return salto, en_suelo

def entrenar_arbol_movimiento(datos_movimiento):
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar el árbol de movimiento.")
        return None

    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32')
    y = datos[:, 8].astype('int') 

    arbol = DecisionTreeClassifier(max_depth=10)
    arbol.fit(X, y)

    return arbol

def decidir_movimiento_arbol(jugador, bala_aire, arbol_movimiento, salto, bala_suelo):
    if arbol_movimiento is None:
        print("Árbol de movimiento no entrenado.")
        return jugador.x, 1 

    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)

    entrada = np.array([[
        jugador.x,
        jugador.y,
        bala_aire.centerx,
        bala_aire.centery,
        bala_suelo.x,
        bala_suelo.y,
        distancia_bala_suelo,
        1 if salto else 0
    ]], dtype='float32')

    accion = arbol_movimiento.predict(entrada)[0]

    if accion == 0 and jugador.x > 0:
        jugador.x -= 5
    elif accion == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
    else:
        jugador.x += 0  

    return jugador.x, accion