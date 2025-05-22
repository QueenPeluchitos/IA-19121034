import pygame, random, tensorflow as tf, numpy as np, pickle, os
from collections import Counter

pygame: motor para crear el juego.
random: genera velocidades aleatorias para las balas.
tensorflow: para el modelo de IA (red neuronal).
numpy: manejo de vectores y matrices para el modelo.
pickle: guardar y cargar datos de entrenamiento.
os: gestión de archivos.
Counter: estadísticas de las acciones tomadas durante el juego.

-Configuración inicial-
pygame.init()
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))

Define la pantalla

-Variables-
jugador_rect = pygame.Rect(50, h - 100, 32, 48)
posicion_inicial_x = jugador_rect.x
bala_rect = pygame.Rect(w - 50, h - 90, 16, 16)
nave_rect = pygame.Rect(w - 100, 290, 64, 64)
nave_rect2 = pygame.Rect(w - 765, 0, 64, 64)
bala_rect2 = pygame.Rect(nave_rect2.centerx - 8, nave_rect2.bottom, 16, 16)

Define jugador, balas y naves.

-Variables de salto y físicas-
salto = False
salto_altura_inicial = 15
salto_altura_actual = salto_altura_inicial
gravedad = 1
en_suelo = True

bala_disparada = False
bala_disparada2 = False

Controlan el salto con parábola.
Controlan si las balas están disparadas o no.

-Modelo de IA-
MODELO_PATH = 'modelo_salto.keras'
modelo = None

Se usa Keras para cargar el archivo de entrenamiento

-Funciones principales-
disparar_bala()
disparar_bala2()
Activan la bala horizontal y vertical con velocidades aleatorias.

reset_bala()
reset_bala2()
Reposicionan las balas al punto de origen cuando salen de la pantalla.

manejar_salto()
Controla el salto del jugador con física de gravedad y parábola.

update_game_state()
Dibuja fondo, jugador, naves y balas.
Actualiza posiciones de las balas.
Detecta colisiones entre jugador y balas.
Si colisión, reinicia el juego y vuelve al menú.

-Recolección de datos-
guardar_datos_para_modelo(teclas)
datos_modelo.append((vel_bala, dist_x, vel_bala2_y, dist_y, accion))

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
    ]])

    accion = modelo.predict(entrada)[0]
    print(f"Acción predicha por KNN: {accion}")

    if accion == 1:
        jugador_rect.x = max(0, jugador_rect.x - velocidad_jugador)
    elif accion == 2:
        jugador_rect.x = min(w - jugador_rect.width, jugador_rect.x + velocidad_jugador)
    elif accion == 3 and en_suelo:
        salto = True
        en_suelo = False

Usa datos actuales del juego para predecir acción con el modelo.
Ejecuta la acción predicha: mover izquierda, derecha o saltar.

-Entrenamiento-
def entrenar_modelo_desde_datos():
    global modelo

    datos_para_entrenamiento = []
    if os.path.exists('datos_entrenamiento.pkl'):
        with open('datos_entrenamiento.pkl', 'rb') as f:
            datos_para_entrenamiento = pickle.load(f)

    datos_para_entrenamiento = [d for d in datos_para_entrenamiento if len(d) == 5]
    datos_para_entrenamiento.extend(datos_modelo)

    if not datos_para_entrenamiento:
        print("No hay datos suficientes.")
        return

    entradas = np.array([[d[0], d[1], d[2], d[3]] for d in datos_para_entrenamiento])
    salidas = np.array([d[4] for d in datos_para_entrenamiento])

    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(entradas, salidas)

    with open(MODELO_PATH, 'wb') as f:
        pickle.dump(modelo, f)
    print("Modelo KNN entrenado y guardado.")

Carga datos anteriores y combina con los nuevos.
Prepara datos de entrada (X) y salida (y).
Entrena una red neuronal simple con TensorFlow (2 capas ocultas).
Guarda el modelo entrenado en disco (modelo_salto.keras).

-Bucle main-
def main():
    global salto, en_suelo, pausa
    reloj = pygame.time.Clock()
    correr = True
    while correr:
        if menu_activo:
            mostrar_menu()

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if not menu_activo and not pausa:
                    if evento.key == pygame.K_SPACE and en_suelo:
                        salto = True
                        en_suelo = False
                    if evento.key == pygame.K_LEFT:
                        jugador_rect.x = max(0, jugador_rect.x - velocidad_jugador)
                    if evento.key == pygame.K_RIGHT:
                        jugador_rect.x = min(w - jugador_rect.width, jugador_rect.x + velocidad_jugador)
                if evento.key == pygame.K_p:
                    pausa = not pausa
                if evento.key == pygame.K_q:
                    correr = False

        teclas = pygame.key.get_pressed()

        if not modo_auto:
            if not teclas[pygame.K_LEFT] and not teclas[pygame.K_RIGHT]:
                if jugador_rect.x > posicion_inicial_x:
                    jugador_rect.x -= 2
                elif jugador_rect.x < posicion_inicial_x:
                    jugador_rect.x += 2

        if not menu_activo and not pausa:
            if modo_auto:
                decision_auto()
            if salto or (not en_suelo and jugador_rect.y < h - 100):
                manejar_salto()
            if not modo_auto:
                guardar_datos_para_modelo(teclas)
            if not bala_disparada:
                disparar_bala()
            if not bala_disparada2:
                disparar_bala2()
            update_game_state()

        pygame.display.flip()
        reloj.tick(30)

    guardar_datos_a_archivo()
    pygame.quit()

Movimiento manual:
Izquierda y derecha con flechas.
Salto con barra espaciadora (si está en suelo).

Si no hay input manual y no está en modo automático, el jugador vuelve poco a poco a su posición inicial.

Si las balas no están activas, se disparan automáticamente con velocidades aleatorias.