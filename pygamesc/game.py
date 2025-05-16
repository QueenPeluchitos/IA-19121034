import pygame
import random
import tensorflow as tf
import numpy as np
import pickle
import os
from collections import Counter

pygame.init()

w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables principales
jugador_rect = pygame.Rect(50, h - 100, 32, 48)
posicion_inicial_x = jugador_rect.x
bala_rect = pygame.Rect(w - 50, h - 90, 16, 16)
nave_rect = pygame.Rect(w - 100, 290, 64, 64)
nave_rect2 = pygame.Rect(w - 765, 0, 64, 64)
bala_rect2 = pygame.Rect(nave_rect2.centerx - 8, nave_rect2.bottom, 16, 16)

salto = False
salto_altura_inicial = 15
salto_altura_actual = salto_altura_inicial
gravedad = 1
en_suelo = True
pausa = False
menu_activo = True
modo_auto = False
velocidad_jugador = 15

bala_disparada = False
bala_disparada2 = False
velocidad_bala_actual = -10
velocidad_bala2_y = 5

fondo_x1 = 0
fondo_x2 = w

fuente_grande = pygame.font.SysFont('Arial', 28)
fuente_media = pygame.font.SysFont('Arial', 24)

datos_modelo = []

try:
    jugador_frames = [
        pygame.image.load('assets/sprites/mono_frame_1.png'),
        pygame.image.load('assets/sprites/mono_frame_2.png'),
        pygame.image.load('assets/sprites/mono_frame_3.png'),
        pygame.image.load('assets/sprites/mono_frame_4.png')
    ]
    bala_img = pygame.image.load('assets/sprites/purple_ball.png')
    fondo_img = pygame.image.load('assets/game/fondo2.png')
    nave_img = pygame.image.load('assets/game/ufo.png')
except pygame.error as e:
    print(f"Error al cargar imágenes: {e}")
    pygame.quit()
    exit()

fondo_img = pygame.transform.scale(fondo_img, (w, h))

MODELO_PATH = 'modelo_salto.keras'
modelo = None
if os.path.exists(MODELO_PATH):
    modelo = tf.keras.models.load_model(MODELO_PATH)
    print("Modelo cargado correctamente.")

def disparar_bala():
    global bala_disparada, velocidad_bala_actual
    if not bala_disparada:
        velocidad_bala_actual = random.randint(-8, -3)
        bala_disparada = True

def disparar_bala2():
    global bala_disparada2, bala_rect2, velocidad_bala2_y
    if not bala_disparada2:
        bala_rect2.x = nave_rect2.centerx - 8
        bala_rect2.y = nave_rect2.bottom
        velocidad_bala2_y = random.randint(4, 7)
        bala_disparada2 = True

def reset_bala():
    global bala_rect, bala_disparada
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
        if jugador_rect.y >= h - 100:
            jugador_rect.y = h - 100
            salto = False
            salto_altura_actual = salto_altura_inicial
            en_suelo = True

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

    datos_modelo.append((
        float(velocidad_bala_actual),
        float(distancia_x),
        float(velocidad_bala2_y),
        float(distancia_y),
        accion
    ))

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

def guardar_datos_a_archivo():
    if datos_modelo:
        with open('datos_entrenamiento.pkl', 'wb') as f:
            pickle.dump(datos_modelo, f)
        acciones = [d[4] for d in datos_modelo]
        print("Acciones guardadas:", dict(Counter(acciones)))

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

def reiniciar_juego_a_menu():
    global menu_activo, salto, en_suelo, bala_disparada, bala_disparada2, salto_altura_actual
    menu_activo = True
    jugador_rect.x, jugador_rect.y = 50, h - 100
    bala_rect.x = w - 50
    bala_rect2.x = jugador_rect.centerx - 8
    bala_rect2.y = nave_rect2.bottom
    bala_disparada = False
    bala_disparada2 = False
    salto = False
    en_suelo = True
    salto_altura_actual = salto_altura_inicial

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
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                guardar_datos_a_archivo()
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a and modelo:
                    modo_auto = True
                    menu_activo = False
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_t:
                    entrenar_modelo_desde_datos()
                elif evento.key == pygame.K_q:
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

if __name__ == "__main__":
    main()