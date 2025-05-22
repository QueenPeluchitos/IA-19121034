-Importaciones-
import pygame, random, numpy as np, pickle, os
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

pygame para la lógica del juego.
random para generar valores aleatorios.
numpy para manipular arrays.
pickle para guardar y cargar el modelo IA y los datos.
os para verificar archivos.
Counter para contar acciones guardadas.
DecisionTreeClassifier para crear el modelo IA.

-Configuración inicial-
pygame.init()
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))

Define la pantalla

-Inicialización de elementos del juego-
jugador_rect = pygame.Rect(50, h - 100, 32, 48)
posicion_inicial_x = jugador_rect.x
bala_rect = pygame.Rect(w - 50, h - 90, 16, 16)
nave_rect = pygame.Rect(w - 100, 290, 64, 64)
nave_rect2 = pygame.Rect(w - 765, 0, 64, 64)
bala_rect2 = pygame.Rect(nave_rect2.centerx - 8, nave_rect2.bottom, 16, 16)

Se manejan variables como salto, gravedad, bala_disparada, etc., para controlar físicas y estado de disparos.

-Cargar modelo de IA-
if os.path.exists(MODELO_PATH):
    modelo_arbol = pickle.load(f)

-Funciones para disparo de balas-
disparar_bala()
disparar_bala2() 

Disparan proyectiles desde la derecha o desde una nave superior.

reset_bala()
reset_bala2() 
Reinician la posición de las balas al salir de la pantalla.

-Función para manejar el salto-
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

Mueve al jugador hacia arriba restando una altura decreciente. Al alcanzar el suelo, reinicia el estado.

-Recolección de datos-
def guardar_datos_para_modelo(teclas):
    distancia_x = abs(jugador_rect.x - bala_rect.x)
    distancia_y = abs(jugador_rect.y - bala_rect2.y)

    accion = 0  # quieto
    if salto:
        accion = 3
    elif teclas[pygame.K_LEFT]:
        accion = 1
    elif teclas[pygame.K_RIGHT]:
        accion = 2

Almacena distancia entre el jugador y los proyectiles junto con la acción ejecutada:
0: Quieto
1: Izquierda
2: Derecha
3: Salto

-Modo automático-
def decision_auto():
    global salto, en_suelo
    if not modelo_arbol:
        return

    entrada = np.array([[
        float(velocidad_bala_actual),
        float(abs(jugador_rect.x - bala_rect.x)),
        float(velocidad_bala2_y),
        float(abs(jugador_rect.y - bala_rect2.y))
    ]])

    accion = modelo_arbol.predict(entrada)[0]
    print(f"Acción predicha: {accion}")

    if accion == 1:
        jugador_rect.x = max(0, jugador_rect.x - velocidad_jugador)
    elif accion == 2:
        jugador_rect.x = min(w - jugador_rect.width, jugador_rect.x + velocidad_jugador)
    elif accion == 3 and en_suelo:
        salto = True
        en_suelo = False

Usa el modelo de árbol de decisión para predecir la mejor acción

-Entrenamiento-
def entrenar_modelo_desde_datos():
    global modelo_arbol

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

    X = np.array([[d[0], d[1], d[2], d[3]] for d in datos_para_entrenamiento])
    y = np.array([d[4] for d in datos_para_entrenamiento])

    modelo_arbol = DecisionTreeClassifier()
    modelo_arbol.fit(X, y)

    with open(MODELO_PATH, 'wb') as f:
        pickle.dump(modelo_arbol, f)
    print("Modelo de árbol entrenado y guardado.")

Carga todos los datos del archivo y entrena un nuevo modelo DecisionTreeClassifier, que se guarda en disco para usos futuros.

-Bucle, main-
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
                        salto_altura_actual = salto_altura_inicial
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

Gestión de eventos (teclado, pausa, salida).
Lógica de movimiento y salto.
Llamadas a las funciones de IA, actualización gráfica y recolección de datos.

