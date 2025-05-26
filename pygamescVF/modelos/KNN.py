import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def entrenar_knn_salto(datos_modelo):
    if len(datos_modelo) < 10:
        print("Insuficientes datos para entrenar el modelo KNN de salto.")
        return None

    datos = np.array(datos_modelo)
    X = datos[:, :6] 
    y = datos[:, 6]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_knn = KNeighborsClassifier(n_neighbors=10)
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

    return salto, en_suelo

def entrenar_knn_movimiento(datos_movimiento, jugador):
    if len(datos_movimiento) < 10:
        print("Insuficientes datos para entrenar el modelo KNN de movimiento.")
        return jugador.x, 1

    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32') 
    y = datos[:, 8].astype('int') 

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
    elif accion == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
    else:
        jugador.x = jugador.x

    return jugador.x, accion

