# Esta función entrena un árbol de decisión para decidir cuándo saltar
def entrenar_arbol_salto(datos_modelo):
    # Si no tenemos suficientes datos, no vale la pena entrenar
    if len(datos_modelo) < 10:
        print("No hay suficientes datos para entrenar el árbol.")
        return None
    
    # Convertimos los datos a numpy array para que sklearn los entienda
    datos = np.array(datos_modelo)
    X = datos[:, :6]  # Las primeras 6 columnas son las características (features)
    y = datos[:, 6]   # La séptima columna es lo que queremos predecir (saltar o no)
    
    # Dividimos los datos en entrenamiento y prueba (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creamos el árbol de decisión (max_depth=10 para que no se vuelva muy complejo)
    arbol = DecisionTreeClassifier(max_depth=10)
    arbol.fit(X_train, y_train)  # Aquí es donde aprende
    
    # Vemos qué tan bien lo hizo con los datos de prueba
    accuracy = arbol.score(X_test, y_test)
    print(f"Precisión del árbol de salto: {accuracy:.2f}")
    
    return arbol

# Esta función usa el árbol entrenado para decidir si saltar o no
def decidir_salto_arbol(jugador, bala, velocidad_bala, bala_aire, bala_disparada_aire, arbol_salto, salto, en_suelo):
    # Si no hay árbol entrenado, no podemos hacer nada
    if arbol_salto is None:
        print("Árbol no entrenado.")
        return salto, en_suelo
    
    # Calculamos las distancias y características del momento actual
    distancia_suelo = abs(jugador.x - bala.x)  # Qué tan lejos está la bala del suelo
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)  # Distancia horizontal a bala aérea
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)  # Distancia vertical a bala aérea
    hay_bala_aire = 1 if bala_disparada_aire else 0  # Si hay bala en el aire (1) o no (0)
    
    # Creamos el vector de entrada con todas las características
    entrada = np.array([[velocidad_bala, distancia_suelo, distancia_aire_x, distancia_aire_y, hay_bala_aire, jugador.x]])
    
    # Le preguntamos al árbol qué hacer (0 = no saltar, 1 = saltar)
    prediccion = arbol_salto.predict(entrada)[0]
    
    # Si dice que saltemos Y estamos en el suelo, entonces saltamos
    if prediccion == 1 and en_suelo:
        salto = True
        en_suelo = False
    
    return salto, en_suelo

# Esta función entrena un árbol para decidir el movimiento horizontal (izquierda/derecha/quieto)
def entrenar_arbol_movimiento(datos_movimiento):
    # Otra vez, necesitamos datos suficientes
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar el árbol de movimiento.")
        return None
    
    # Preparamos los datos (8 características + 1 acción)
    datos = np.array(datos_movimiento)
    X = datos[:, :8].astype('float32')  # Las primeras 8 columnas
    y = datos[:, 8].astype('int')       # La novena columna es la acción (0=izq, 1=quieto, 2=der)
    
    # Creamos y entrenamos el árbol
    arbol = DecisionTreeClassifier(max_depth=10)
    arbol.fit(X, y)
    
    return arbol

# Esta función usa el árbol de movimiento para decidir hacia dónde moverse
def decidir_movimiento_arbol(jugador, bala_aire, arbol_movimiento, salto, bala_suelo):
    # Sin árbol entrenado, no hacemos nada
    if arbol_movimiento is None:
        print("Árbol de movimiento no entrenado.")
        return jugador.x, 1  # Devolvemos posición actual y acción "quieto"
    
    # Calculamos la distancia a la bala del suelo
    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)
    
    # Preparamos todas las características para el árbol
    entrada = np.array([[
        jugador.x,                    # Posición X del jugador
        jugador.y,                    # Posición Y del jugador
        bala_aire.centerx,            # Posición X de la bala aérea
        bala_aire.centery,            # Posición Y de la bala aérea
        bala_suelo.x,                 # Posición X de la bala del suelo
        bala_suelo.y,                 # Posición Y de la bala del suelo
        distancia_bala_suelo,         # Distancia calculada
        1 if salto else 0             # Si está saltando (1) o no (0)
    ]], dtype='float32')
    
    # Preguntamos al árbol qué hacer (0=izquierda, 1=quieto, 2=derecha)
    accion = arbol_movimiento.predict(entrada)[0]
    
    # Ejecutamos la acción pero con límites para no salirse de la pantalla
    if accion == 0 and jugador.x > 0:  # Moverse a la izquierda
        jugador.x -= 5
    elif accion == 2 and jugador.x < 200 - jugador.width:  # Moverse a la derecha
        jugador.x += 5
    else:  # Quedarse quieto (o si está en el borde)
        jugador.x += 0 
    
    return jugador.x, accion