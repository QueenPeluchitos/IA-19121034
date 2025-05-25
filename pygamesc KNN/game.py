import pygame
import random
import pickle
import os
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

pygame.init()

w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("K-Vecinos más Cercanos")

BLANCO = (255, 255, 255)
NEGRO  = (0, 0, 0)

# jugador y posición inicial
jugador_rect      = pygame.Rect(50, h - 100, 32, 48)
posicion_inicial_x = jugador_rect.x
GROUND_Y           = jugador_rect.y

# límite para avanzar hacia la derecha: 100 px desde la posición inicial
LIMITE_DERECHA = posicion_inicial_x + 100

# enemigos y balas
bala_rect   = pygame.Rect(w - 50, h - 90, 16, 16)
nave_rect   = pygame.Rect(w - 100, 290, 64, 64)
nave_rect2  = pygame.Rect(w - 765, 0, 64, 64)
bala_rect2  = pygame.Rect(nave_rect2.centerx - 8, nave_rect2.bottom, 16, 16)
accion_anterior = 0


# estados de salto
salto                 = False
salto_altura_inicial  = 15
salto_altura_actual   = salto_altura_inicial
gravedad              = 1
en_suelo              = True

pausa       = False
menu_activo = True
modo_auto   = False

# velocidad del jugador (solo para movimiento en cada frame)
velocidad_jugador = 15

movimiento_activo    = False
movimiento_direccion = 0   
movimiento_phase     = 'forward'
movimiento_steps     = 0

# balas
bala_disparada   = False
bala_disparada2  = False
velocidad_bala_actual = -10
velocidad_bala2_y     = 5

# fuentes
fuente_grande = pygame.font.SysFont('Arial', 28)
fuente_media  = pygame.font.SysFont('Arial', 24)

# datos para entrenar y modelo KNN
datos_modelo = []
modelo       = None
MODELO_PATH  = 'modelo_knn.pkl'

# carga de imágenes
try:
    jugador_frames = [pygame.image.load(f'assets/sprites/mono_frame_{i}.png') for i in range(1,5)]
    bala_img        = pygame.image.load('assets/sprites/purple_ball.png')
    fondo_img       = pygame.image.load('assets/game/fondo2.png')
    nave_img        = pygame.image.load('assets/game/ufo.png')
except pygame.error as e:
    print(f"Error al cargar imágenes: {e}")
    pygame.quit()
    exit()

fondo_img = pygame.transform.scale(fondo_img, (w, h))

# carga de modelo si existe
if os.path.exists(MODELO_PATH):
    with open(MODELO_PATH, 'rb') as f:
        modelo = pickle.load(f)
    print("Modelo KNN cargado correctamente.")

# --- funciones de juego ---
def disparar_bala():
    global bala_disparada, velocidad_bala_actual
    if not bala_disparada:
        velocidad_bala_actual = random.randint(-8, -3)
        bala_disparada = True

def disparar_bala2():
    global bala_disparada2, velocidad_bala2_y
    if not bala_disparada2:
        bala_rect2.x = nave_rect2.centerx - 8
        bala_rect2.y = nave_rect2.bottom
        velocidad_bala2_y = random.randint(4, 7)
        bala_disparada2 = True

def reset_bala():
    global bala_disparada
    bala_rect.x = w - 50
    bala_disparada = False

def reset_bala2():
    global bala_disparada2
    bala_disparada2 = False

def manejar_salto():
    global salto, salto_altura_actual, en_suelo
    if salto:
        jugador_rect.y -= salto_altura_actual
        salto_altura_actual -= gravedad
        if jugador_rect.y >= GROUND_Y:
            jugador_rect.y = GROUND_Y
            salto = False
            salto_altura_actual = salto_altura_inicial
            en_suelo = True

def iniciar_movimiento(direccion):
    global movimiento_activo, movimiento_direccion
    movimiento_activo = True
    movimiento_direccion = direccion

def detener_movimiento():
    global movimiento_activo
    movimiento_activo = False

def manejar_movimiento_lateral():
    global movimiento_activo
    if not movimiento_activo:
        return

    paso = velocidad_jugador
    # Limitar movimiento dentro de la pantalla
    nueva_x = jugador_rect.x + movimiento_direccion * paso
    nueva_x = max(0, min(w - jugador_rect.width, nueva_x))
    jugador_rect.x = nueva_x
def update_game_state():
    pantalla.blit(fondo_img, (0, 0))
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
        reiniciar_juego_a_menu()

def guardar_datos_para_modelo(teclas):
    dx = abs(jugador_rect.x - bala_rect.x)
    dy = abs(jugador_rect.y - bala_rect2.y)
    accion = 0
    if salto:
        accion = 3
    elif teclas[pygame.K_LEFT]:
        accion = 1
    elif teclas[pygame.K_RIGHT]:
        accion = 2
    datos_modelo.append((velocidad_bala_actual, dx, velocidad_bala2_y, dy, accion))

def decision_auto():
    global salto, en_suelo, accion_anterior, movimiento_activo
    if not modelo:
        return
    entrada = np.array([[velocidad_bala_actual,
                         abs(jugador_rect.x - bala_rect.x),
                         velocidad_bala2_y,
                         abs(jugador_rect.y - bala_rect2.y)]])
    accion = modelo.predict(entrada)[0]
    # Solo cambiar movimiento si la acción cambia
    if accion != accion_anterior:
        if accion == 0:
            # detener movimiento
            movimiento_activo = False
        elif accion == 1:
            iniciar_movimiento(-1)
        elif accion == 2:
            iniciar_movimiento(+1)
        elif accion == 3 and en_suelo:
            salto = True
            en_suelo = False
        accion_anterior = accion

def entrenar_modelo_desde_datos():
    global modelo
    datos = []
    if os.path.exists('datos_entrenamiento.pkl'):
        with open('datos_entrenamiento.pkl', 'rb') as f:
            datos = pickle.load(f)
    datos.extend(datos_modelo)
    if not datos:
        print("No hay datos suficientes.")
        return
    X = np.array([[d[0], d[1], d[2], d[3]] for d in datos])
    y = np.array([d[4] for d in datos])
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X, y)
    with open(MODELO_PATH, 'wb') as f:
        pickle.dump(modelo, f)
    print("Modelo KNN entrenado y guardado.")

def reiniciar_juego_a_menu():
    global menu_activo, salto, en_suelo, bala_disparada, bala_disparada2, salto_altura_actual
    global movimiento_activo, movimiento_steps
    jugador_rect.x, jugador_rect.y = posicion_inicial_x, GROUND_Y
    salto = False
    en_suelo = True
    salto_altura_actual = salto_altura_inicial
    bala_rect.x = w - 50
    bala_rect2.x, bala_rect2.y = nave_rect2.centerx - 8, nave_rect2.bottom
    bala_disparada = False
    bala_disparada2 = False
    movimiento_activo = False
    movimiento_steps = 0
    menu_activo = True

def mostrar_menu():
    global menu_activo, modo_auto
    while menu_activo:
        pantalla.fill(NEGRO)
        pantalla.blit(fuente_grande.render("MENU PRINCIPAL", True, BLANCO), (w//2 - 120, h//4))
        pantalla.blit(fuente_media.render("Presiona 'A' para Modo Automático", True, BLANCO), (w//2 - 180, h//2 - 30))
        pantalla.blit(fuente_media.render("Presiona 'M' para Modo Manual", True, BLANCO), (w//2 - 160, h//2))
        pantalla.blit(fuente_media.render("Presiona 'T' para Entrenar Modelo", True, BLANCO), (w//2 - 180, h//2 + 30))
        pantalla.blit(fuente_media.render("Presiona 'Q' para Salir", True, BLANCO), (w//2 - 120, h//2 + 60))
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                guardar_datos_a_archivo()
                pygame.quit()
                exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_a and modelo:
                    modo_auto = True
                    menu_activo = False
                elif e.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                elif e.key == pygame.K_t:
                    entrenar_modelo_desde_datos()
                elif e.key == pygame.K_q:
                    guardar_datos_a_archivo()
                    pygame.quit()
                    exit()

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
            if evento.type == pygame.KEYDOWN and not menu_activo and not pausa:
                if evento.key == pygame.K_SPACE and en_suelo:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_LEFT:
                    iniciar_movimiento(-1)
                if evento.key == pygame.K_RIGHT:
                    iniciar_movimiento(+1)
                if evento.key == pygame.K_p:
                    pausa = not pausa
                if evento.key == pygame.K_q:
                    correr = False
            if evento.type == pygame.KEYUP and not menu_activo and not pausa:
                if evento.key == pygame.K_LEFT and movimiento_direccion == -1:
                    detener_movimiento()
                if evento.key == pygame.K_RIGHT and movimiento_direccion == +1:
                    detener_movimiento()


        teclas = pygame.key.get_pressed()
        if not modo_auto and not menu_activo and not pausa:
            guardar_datos_para_modelo(teclas)
        if modo_auto and not menu_activo and not pausa:
            decision_auto()

        if salto or (not en_suelo and jugador_rect.y < GROUND_Y):
            manejar_salto()

        manejar_movimiento_lateral()

        if not menu_activo and not pausa:
            if not bala_disparada:
                disparar_bala()
            if not bala_disparada2:
                disparar_bala2()
            update_game_state()

        pygame.display.flip()
        reloj.tick(30)

    guardar_datos_a_archivo()
    pygame.quit()

if __name__ == "__main__":
    main()