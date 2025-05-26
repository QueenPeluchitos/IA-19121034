# Proyecto 1: Phaser en Python

__ Este proyecto se utilizan 3 tipos de modelo de entrenamiento para jugar automaticamente con aprendizaje automatico __ 

## RED NEURONAL 
```python
import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import to_categorical


# ---------------- MODELO DE RED NEURONAL ----------------
# ---------------- RED NEURONAL DE SALTO -----------------
def entrenar_modelo(datos_modelo):
    if len(datos_modelo) < 10:
        print("Insuficientes datos para entrenar el modelo.")
        return None

    datos = np.array(datos_modelo)
    X = datos[:, :6]  # Nuevas 6 características
    y = datos[:, 6]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = Sequential([
        Dense(32, input_dim=6, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    modelo.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"Modelo entrenado con precisión: {accuracy:.2f}")
    
    return modelo

def decidir_salto(jugador, bala, velocidad_bala, bala_aire, bala_disparada_aire, modelo_entrenado, salto, en_suelo):
    if modelo_entrenado is None:
        print("Modelo no entrenado. No se puede decidir.")
        return False, en_suelo

    distancia_suelo = abs(jugador.x - bala.x)
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)
    hay_bala_aire = 1 if bala_disparada_aire else 0

    entrada = np.array([[velocidad_bala, distancia_suelo, distancia_aire_x, distancia_aire_y, hay_bala_aire, jugador.x]])

    prediccion = modelo_entrenado.predict(entrada, verbose=0)[0][0]

    if prediccion > 0.5 and en_suelo:
        salto = True
        en_suelo = False
        print("Saltar")

    return salto, en_suelo


 
# ---------------- RED NEURONAL DE MOVIMIENTO -----------------
def entrenar_red_movimiento(datos_movimiento):
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar.")
        return None

    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32')  # 8 entradas
    y = datos[:, 8].astype('int')       # Acción: izquierda, quieto, derecha

    y_categorical = to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, input_dim=8, activation='relu'),  # Cambiado a 8 entradas
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión del modelo de movimiento: {accuracy:.2f}")
    
    return model

def decidir_movimiento(jugador, bala, modelo_movimiento, salto, bala_suelo):
    if modelo_movimiento is None:
        print("Modelo no entrenado.")
        return jugador.x, 1  # Quieto por defecto

    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)

    entrada = np.array([[
        jugador.x,                     # Posición actual del jugador
        jugador.y,                     # Altura del jugador
        bala.centerx,                  # X de la bala aérea
        bala.centery,                  # Y de la bala aérea
        bala_suelo.x,                  # X de la bala del suelo
        bala_suelo.y,                  # Y de la bala del suelo
        distancia_bala_suelo,          # Distancia a bala del suelo
        1 if salto else 0              # Si el jugador está saltando
    ]], dtype='float32')

    prediccion = modelo_movimiento.predict(entrada, verbose=0)[0]
    accion = np.argmax(prediccion)

    if accion == 0 and jugador.x > 0:
        jugador.x -= 5
        print("Izquierda")
    elif accion == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
        print("Derecha")
    else:
        print("Quieto")

    return jugador.x, accion


```

## ARBOL DE DECISIÓN
```python
import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------------- ARBOL DE DECISION ----------------


def entrenar_arbol_salto(datos_modelo):
    if len(datos_modelo) < 10:
        print("No hay suficientes datos para entrenar el árbol.")
        return None

    datos = np.array(datos_modelo)
    X = datos[:, :6]  # 6 características
    y = datos[:, 6]   # salto_hecho (0 o 1)

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
        print("Saltar")

    return salto, en_suelo



def entrenar_arbol_movimiento(datos_movimiento):
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar el árbol de movimiento.")
        return None

    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32')
    y = datos[:, 8].astype('int')  # 0=izquierda, 1=quieto, 2=derecha

    arbol = DecisionTreeClassifier(max_depth=10)
    arbol.fit(X, y)

    return arbol



def decidir_movimiento_arbol(jugador, bala_aire, arbol_movimiento, salto, bala_suelo):
    if arbol_movimiento is None:
        print("Árbol de movimiento no entrenado.")
        return jugador.x, 1  # Quieto

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
        print("Izquierda")
    elif accion == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
        print("Derecha (árbol)")
    else:
        print("Quieto (árbol)")

    return jugador.x, accion
```


## K NEAREST NEIGHBOURS
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

# ----------- KNN PARA DECIDIR SALTO ------------
def entrenar_knn_salto(datos_modelo):
    if len(datos_modelo) < 10:
        print("Insuficientes datos para entrenar el modelo KNN de salto.")
        return None

    datos = np.array(datos_modelo)
    X = datos[:, :6]  # 6 características: velocidad_bala, distancia_suelo, etc.
    y = datos[:, 6]   # Etiqueta: salto (0 o 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_knn = KNeighborsClassifier(n_neighbors=10)  # Puedes ajustar n_neighbors
    modelo_knn.fit(X_train, y_train)

    accuracy = modelo_knn.score(X_test, y_test)
    print(f"Precisión del modelo KNN de salto: {accuracy:.2f}")
    
    return modelo_knn

def decidir_salto_knn(jugador, bala, velocidad_bala, bala_aire, bala_disparada_aire, modelo_knn, salto, en_suelo):
    if modelo_knn is None:
        print("Modelo KNN no entrenado. No se puede decidir.")
        return False, en_suelo

    distancia_suelo = abs(jugador.x - bala.x)
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)
    hay_bala_aire = 1 if bala_disparada_aire else 0

    entrada = np.array([[velocidad_bala, distancia_suelo, distancia_aire_x, distancia_aire_y, hay_bala_aire, jugador.x]])

    prediccion = modelo_knn.predict(entrada)[0]

    if prediccion == 1 and en_suelo:
        salto = True
        en_suelo = False
        print("Saltar (KNN)")

    return salto, en_suelo

def entrenar_knn_movimiento(datos_movimiento):
    if len(datos_movimiento) < 10:
        print("Insuficientes datos para entrenar el modelo KNN de movimiento.")
        return jugador.x, 1

    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32')  # 8 características
    y = datos[:, 8].astype('int')       # 0: izquierda, 1: quieto, 2: derecha

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_knn = KNeighborsClassifier(n_neighbors=10)
    modelo_knn.fit(X_train, y_train)

    accuracy = modelo_knn.score(X_test, y_test)
    print(f"Precisión del modelo KNN de movimiento: {accuracy:.2f}")

    return modelo_knn

def decidir_movimiento_knn(jugador, bala_aire, modelo_knn_mov, salto, bala_suelo):
    if modelo_knn_mov is None:
        print("Modelo KNN de movimiento no entrenado.")
        return None

    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)

    entrada = np.array([[jugador.x,
                         jugador.y,
                         bala_aire.centerx,
                         bala_aire.centery,
                         bala_suelo.x,
                         bala_suelo.y,
                         distancia_bala_suelo,
                         1 if salto else 0
                         ]], dtype='float32')

    prediccion = modelo_knn_mov.predict(entrada)
    accion = int(prediccion[0])

    if accion == 0 and jugador.x > 0:
        jugador.x -= 5
        print("Izquierda (KNN)")
    elif accion == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
        print("Derecha (KNN)")
    else:
        print("Quieto (KNN)")

    return jugador.x, accion

```