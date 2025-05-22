-Importaciones-
import pygame, random, tensorflow as tf, numpy as np, pickle, os
from collections import Counter

pygame: motor del juego.
random: velocidades aleatorias de proyectiles.
tensorflow: modelo de IA.
numpy: para vectores/matrices del modelo.
pickle: guardar y cargar datos de entrenamiento.
os: gestión de archivos.
Counter: estadísticas de acciones tomadas.

-Configuración inicial-
pygame.init()
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))

Define la pantalla

-Variables-
jugador_rect = pygame.Rect(50, h - 100, 32, 48)
bala_rect = pygame.Rect(w - 50, h - 90, 16, 16)
nave_rect = pygame.Rect(w - 100, 290, 64, 64)
nave_rect2 = pygame.Rect(w - 765, 0, 64, 64)
bala_rect2 = pygame.Rect(nave_rect2.centerx - 8, nave_rect2.bottom, 16, 16)

Jugador, balas, naves

-Variables de salto y físicas-
salto = False
salto_altura_inicial = 15
salto_altura_actual = salto_altura_inicial
gravedad = 1
en_suelo = True
bala_disparada = False
bala_disparada2 = False

Salto y balas

-Modelo-
MODELO_PATH = 'modelo_salto.keras'
modelo = None

Se usa Keras para cargar el archivo de entrenamiento

-Funciones-
disparar_bala() 
disparar_bala2()

Activan la bala horizontal y vertical con velocidades aleatorias.

reset_bala()
reset_bala2()

Reposicionan las balas al punto de origen cuando salen de la pantalla.

def manejar_salto():
    global salto, salto_altura_actual, en_suelo
    if salto:
        jugador_rect.y -= salto_altura_actual
        salto_altura_actual -= gravedad
        if jugador_rect.y >= h - 100:
            jugador_rect.y = h - 100
            salto = False
            salto_altura_actual = salto_altura_inicial
            en_suelo = True

Controla el salto del jugador usando una parábola (velocidad disminuye por gravedad hasta tocar el suelo).

def update_game_state():
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))
    pantalla.blit(jugador_frames[0], jugador_rect)
    pantalla.blit(nave_img, nave_rect)
    pantalla.blit(nave_img, nave_rect2)

    if bala_disparada:
        bala_rect.x += velocidad_bala_actual
        if bala_rect.x < 0:
            reset_bala()
    pantalla.blit(bala_img, bala_rect)

    if bala_disparada2:
        bala_rect2.y += velocidad_bala2_y
        if bala_rect2.y > h:
            reset_bala2()
    pantalla.blit(bala_img, bala_rect2)

    if jugador_rect.colliderect(bala_rect) or jugador_rect.colliderect(bala_rect2):
        print("¡Colisión detectada!")
        reiniciar_juego_a_menu()

Dibuja fondo, jugador, naves y balas.
Mueve las balas.
Detecta colisiones.
Si hay colisión, reinicia el juego y regresa al menú.

-Recolección de datos-
guardar_datos_para_modelo(teclas)
datos_modelo.append((vel_bala, dist_x, vel_bala2_y, dist_y, accion))

Guarda:
Velocidad de las balas.
Distancias entre el jugador y las balas.
Acción tomada (0 = nada, 1 = izquierda, 2 = derecha, 3 = salto).

-Modo automático-
def decision_auto():
    global salto, en_suelo
    if not modelo:
        return

    entrada = np.array([[ 
        float(velocidad_bala_actual),
        float(abs(jugador_rect.x - bala_rect.x)),
        float(velocidad_bala2_y),
        float(abs(jugador_rect.y - bala_rect2.y))
    ]], dtype=np.float32)

    prediccion = modelo.predict(entrada, verbose=0)[0]
    accion = np.argmax(prediccion)
    print(f"Predicción: {prediccion} → Acción: {accion}")

    if accion == 1:
        jugador_rect.x = max(0, jugador_rect.x - velocidad_jugador)
    elif accion == 2:
        jugador_rect.x = min(w - jugador_rect.width, jugador_rect.x + velocidad_jugador)
    elif accion == 3 and en_suelo:
        salto = True
        en_suelo = False

Toma los datos actuales del juego.
Usa el modelo IA para predecir qué hacer.
Aplica esa acción: mover izquierda, derecha o saltar.

-Entrenar modelo-
def entrenar_modelo_desde_datos():
    global modelo

    if os.path.exists(MODELO_PATH):
        print("Eliminando modelo antiguo...")
        os.remove(MODELO_PATH)

    datos_para_entrenamiento = []
    try:
        with open('datos_entrenamiento.pkl', 'rb') as f:
            datos_para_entrenamiento = pickle.load(f)
    except FileNotFoundError:
        print("Archivo de datos no encontrado.")

    datos_para_entrenamiento = [d for d in datos_para_entrenamiento if len(d) == 5]
    datos_para_entrenamiento.extend(datos_modelo)

    if not datos_para_entrenamiento:
        print("No hay datos suficientes.")
        return

    entradas = np.array([[d[0], d[1], d[2], d[3]] for d in datos_para_entrenamiento], dtype=np.float32)
    salidas = np.array([d[4] for d in datos_para_entrenamiento], dtype=np.int32)

    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    modelo.fit(entradas, salidas, epochs=50, batch_size=32, validation_split=0.2)
    modelo.save(MODELO_PATH)
    print("Modelo entrenado y guardado.")

Carga los datos del archivo y combina con los nuevos.
Prepara entradas (X) y salidas (y).
Entrena un modelo de red neuronal simple con 2 capas ocultas.
Guarda el modelo en disco.

-Bucle main-
if evento.key == pygame.K_LEFT:
    jugador_rect.x = max(0, jugador_rect.x - velocidad_jugador)
if evento.key == pygame.K_RIGHT:
    jugador_rect.x = min(w - jugador_rect.width, jugador_rect.x + velocidad_jugador)

Salto con la barra espaciadora

if evento.key == pygame.K_LEFT:
    jugador_rect.x = max(0, jugador_rect.x - velocidad_jugador)
if evento.key == pygame.K_RIGHT:
    jugador_rect.x = min(w - jugador_rect.width, jugador_rect.x + velocidad_jugador)

Movimiento manual del jugador

if not modo_auto:
    if not teclas[pygame.K_LEFT] and not teclas[pygame.K_RIGHT]:
        if jugador_rect.x > posicion_inicial_x:
            ugador_rect.x -= 2
        elif jugador_rect.x < posicion_inicial_x:
            jugador_rect.x += 2

Movimiento automático hacia la posición inicial

if not bala_disparada:
    disparar_bala()
if not bala_disparada2:
    disparar_bala2()
    
Disparo de balas



