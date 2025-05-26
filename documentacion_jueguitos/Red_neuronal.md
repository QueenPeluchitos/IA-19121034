# Esta función entrena una red neuronal para decidir cuándo saltar
def entrenar_modelo(datos_modelo):
    # Necesitamos suficientes datos para que la red neuronal aprenda bien
    if len(datos_modelo) < 10:
        print("Insuficientes datos para entrenar el modelo.")
        return None

    # Preparamos los datos como siempre
    datos = np.array(datos_modelo)
    X = datos[:, :6]  # Las primeras 6 características (velocidad, distancias, etc.)
    y = datos[:, 6]   # La decisión de saltar (0 o 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos la red neuronal 
    modelo = Sequential([
        Dense(32, input_dim=6, activation='relu'),  # Primera capa: 32 neuronas, recibe 6 entradas
        Dense(32, activation='relu'),               # Segunda capa: 32 neuronas más
        Dense(16, activation='relu'),               # Tercera capa: 16 neuronas (se va reduciendo)
        Dense(1, activation='sigmoid')              # Última capa: 1 neurona que dice sí/no (0-1)
    ])
    
    # Le decimos cómo aprender: adam es un optimizador inteligente, binary_crossentropy para sí/no
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Aquí es donde realmente aprende - 100 veces repasa todos los datos
    modelo.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    
    # Probamos qué tan bien aprendió con datos que nunca vio
    loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"Modelo entrenado con precisión: {accuracy:.2f}")
    
    return modelo

# Esta función usa la red neuronal entrenada para decidir si saltar
def decidir_salto(jugador, bala, velocidad_bala, bala_aire, bala_disparada_aire, modelo_entrenado, salto, en_suelo):
    # Sin modelo entrenado, no podemos hacer nada
    if modelo_entrenado is None:
        print("Modelo no entrenado. No se puede decidir.")
        return False, en_suelo

    # Calculamos todas las características del momento actual
    distancia_suelo = abs(jugador.x - bala.x)                    # Distancia a bala del suelo
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)  # Distancia horizontal a bala aérea
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)  # Distancia vertical a bala aérea
    hay_bala_aire = 1 if bala_disparada_aire else 0              # Si hay bala aérea o no

    # Armamos el vector de entrada para la red neuronal
    entrada = np.array([[velocidad_bala, distancia_suelo, distancia_aire_x, distancia_aire_y, hay_bala_aire, jugador.x]])

    # La red neuronal nos da un número entre 0 y 1 (probabilidad de saltar)
    prediccion = modelo_entrenado.predict(entrada, verbose=0)[0][0]

    # Si la probabilidad es mayor a 0.5 (50%) Y estamos en el suelo, saltamos
    if prediccion > 0.5 and en_suelo:
        salto = True
        en_suelo = False
    return salto, en_suelo

# Esta función entrena una red neuronal para decidir el movimiento (izq/quieto/der)
def entrenar_red_movimiento(datos_movimiento):
    # Otra vez, necesitamos datos suficientes
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar.")
        return None

    # Preparamos los datos de movimiento
    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32')  # 8 características
    y = datos[:, 8].astype('int')       # Acción tomada (0, 1, o 2)

    # Convertimos las acciones a formato "categórico" - la red neuronal lo entiende mejor así
    # En lugar de [0,1,2] usamos [[1,0,0], [0,1,0], [0,0,1]]
    y_categorical = to_categorical(y, num_classes=3)

    # Dividimos los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # Creamos la red neuronal para movimiento
    model = Sequential([
        Dense(64, input_dim=8, activation='relu'),  # Primera capa: 64 neuronas, recibe 8 entradas
        Dense(32, activation='relu'),               # Segunda capa: 32 neuronas
        Dense(3, activation='softmax')              # Última capa: 3 neuronas (izquierda/quieto/derecha)
    ])

    # Configuramos el aprendizaje - categorical_crossentropy para múltiples opciones
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Entrenamos la red neuronal
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    # Vemos qué tan bien aprendió
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión del modelo de movimiento: {accuracy:.2f}")
    
    return model

# Esta función usa la red neuronal para decidir el movimiento
def decidir_movimiento(jugador, bala, modelo_movimiento, salto, bala_suelo):
    # Sin modelo, no hay decisión
    if modelo_movimiento is None:
        print("Modelo no entrenado.")
        return jugador.x, 1  # Devolvemos posición actual y acción "quieto"

    # Calculamos la distancia a la bala del suelo
    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)

    # Preparamos todas las características para la red
    entrada = np.array([[
        jugador.x,                    # Posición X del jugador
        jugador.y,                    # Posición Y del jugador
        bala.centerx,                 # Posición X de la bala aérea
        bala.centery,                 # Posición Y de la bala aérea
        bala_suelo.x,                 # Posición X de la bala del suelo
        bala_suelo.y,                 # Posición Y de la bala del suelo
        distancia_bala_suelo,         # Distancia calculada
        1 if salto else 0             # Si está saltando o no
    ]], dtype='float32')

    # La red nos da 3 probabilidades [prob_izq, prob_quieto, prob_der]
    prediccion = modelo_movimiento.predict(entrada, verbose=0)[0]
    
    # Elegimos la acción con mayor probabilidad
    accion = np.argmax(prediccion)  # argmax encuentra el índice del valor más alto

    # Ejecutamos la acción elegida, cuidando los límites
    if accion == 0 and jugador.x > 0:  # Moverse a la izquierda
        jugador.x -= 5
    elif accion == 2 and jugador.x < 200 - jugador.width:  # Moverse a la derecha
        jugador.x += 5
    else:  # Quedarse quieto (o si está en el borde)
        return

    return jugador.x, accion