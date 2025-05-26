# Esta función entrena un modelo KNN para decidir cuándo saltar
# KNN funciona diferente a los árboles: busca los 10 casos más parecidos y decide basándose en ellos
def entrenar_knn_salto(datos_modelo):
    # Necesitamos al menos 10 datos para que tenga sentido
    if len(datos_modelo) < 10:
        print("Insuficientes datos para entrenar el modelo KNN de salto.")
        return None

    # Convertimos a numpy para que sklearn pueda trabajar con los datos
    datos = np.array(datos_modelo)
    X = datos[:, :6]  # Las primeras 6 columnas son las características
    y = datos[:, 6]   # La columna 7 es la decisión (saltar=1, no saltar=0)

    # Dividimos en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos el modelo KNN con 10 vecinos (busca los 10 casos más similares)
    modelo_knn = KNeighborsClassifier(n_neighbors=10)
    modelo_knn.fit(X_train, y_train)  # Aquí "aprende" guardando todos los casos

    # Probamos qué tan bien funciona
    accuracy = modelo_knn.score(X_test, y_test)
    print(f"Precisión del modelo KNN de salto: {accuracy:.2f}")
    
    return modelo_knn

# Esta función usa el modelo KNN entrenado para decidir si saltar
def decidir_salto_knn(jugador, bala, velocidad_bala, bala_aire, bala_disparada_aire, modelo_knn, salto, en_suelo):
    # Sin modelo entrenado, no podemos hacer nada
    if modelo_knn is None:
        print("Modelo KNN no entrenado. No se puede decidir.")
        return False, en_suelo

    # Calculamos todas las distancias y características del momento actual
    distancia_suelo = abs(jugador.x - bala.x)  # Qué tan cerca está la bala del suelo
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)  # Distancia horizontal a bala aérea
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)  # Distancia vertical a bala aérea
    hay_bala_aire = 1 if bala_disparada_aire else 0  # 1 si hay bala aérea, 0 si no

    # Armamos el vector con todas las características
    entrada = np.array([[velocidad_bala, distancia_suelo, distancia_aire_x, distancia_aire_y, hay_bala_aire, jugador.x]])

    # El KNN busca los 10 casos más parecidos y decide basándose en qué hicieron la mayoría
    prediccion = modelo_knn.predict(entrada)[0]

    # Si dice que saltemos Y estamos en el suelo, entonces saltamos
    if prediccion == 1 and en_suelo:
        salto = True
        en_suelo = False

    return salto, en_suelo

# Esta función entrena un modelo KNN para decidir el movimiento horizontal
def entrenar_knn_movimiento(datos_movimiento):
    # Necesitamos suficientes datos para entrenar
    if len(datos_movimiento) < 10:
        print("Insuficientes datos para entrenar el modelo KNN de movimiento.")
        return 10, 1  # Nota: esto parece un error, debería devolver None

    # Preparamos los datos de movimiento
    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32')  # 8 características (posiciones, distancias, etc.)
    y = datos[:, 8].astype('int')       # La acción tomada (0=izq, 1=quieto, 2=der)

    # Dividimos los datos para entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos el modelo KNN para movimiento (también con 10 vecinos)
    modelo_knn = KNeighborsClassifier(n_neighbors=10)
    modelo_knn.fit(X_train, y_train)

    # Vemos qué tan bien predice
    accuracy = modelo_knn.score(X_test, y_test)
    print(f"Precisión del modelo KNN de movimiento: {accuracy:.2f}")

    return modelo_knn

# Esta función usa el KNN entrenado para decidir hacia dónde moverse
def decidir_movimiento_knn(jugador, bala_aire, modelo_knn_mov, salto, bala_suelo):
    # Sin modelo, no podemos decidir
    if modelo_knn_mov is None:
        print("Modelo KNN de movimiento no entrenado.")
        return None

    # Calculamos la distancia a la bala del suelo
    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)

    # Preparamos todas las características para el modelo
    entrada = np.array([[jugador.x,                    # Posición X del jugador
                         jugador.y,                    # Posición Y del jugador
                         bala_aire.centerx,            # Posición X de bala aérea
                         bala_aire.centery,            # Posición Y de bala aérea
                         bala_suelo.x,                 # Posición X de bala del suelo
                         bala_suelo.y,                 # Posición Y de bala del suelo
                         distancia_bala_suelo,         # Distancia calculada
                         1 if salto else 0             # Si está saltando o no
                         ]], dtype='float32')

    # El KNN busca situaciones similares y decide qué acción tomar
    prediccion = modelo_knn_mov.predict(entrada)
    accion = int(prediccion[0])  # 0=izquierda, 1=quieto, 2=derecha

    # Ejecutamos la acción pero cuidando no salirse de los límites
    if accion == 0 and jugador.x > 0:  # Moverse a la izquierda
        jugador.x -= 5
    elif accion == 2 and jugador.x < 200 - jugador.width:  # Moverse a la derecha
        jugador.x += 5
    else:  # Quedarse quieto (o si está en el borde)
        jugador.x += 0 

    return jugador.x, accion