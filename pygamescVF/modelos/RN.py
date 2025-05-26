import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


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

    return salto, en_suelo

def entrenar_red_movimiento(datos_movimiento):
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar.")
        return None

    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32') 
    y = datos[:, 8].astype('int')

    y_categorical = to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, input_dim=8, activation='relu'),
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
        return jugador.x, 1 

    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)

    entrada = np.array([[
        jugador.x,                    
        jugador.y,                    
        bala.centerx,                  
        bala.centery,                 
        bala_suelo.x,                 
        bala_suelo.y,                 
        distancia_bala_suelo,         
        1 if salto else 0             
    ]], dtype='float32')

    prediccion = modelo_movimiento.predict(entrada, verbose=0)[0]
    accion = np.argmax(prediccion)

    if accion == 0 and jugador.x > 0:
        jugador.x -= 5
    elif accion == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
    else:
        return

    return jugador.x, accion

