"""
Juego de esquivar balas con IA
=============================
Un jueguito donde Len tiene que esquivar proyectiles y los modelos de IA aprenden a jugarlo.
Incluye red neuronal, árbol de decisión y KNN.
"""

import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from modelo_red import *
from arbol_decision import *
from knn import * 
import os

# Configuración básica
BASE_DIR = os.path.dirname(__file__)

def cargar_imagen(rel_path):
    """Carga imágenes y maneja errores"""
    ruta = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(ruta):
        print(f"Archivo no encontrado: {ruta}")
    return pygame.image.load(ruta)

pygame.init()

# Pantalla y colores
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Jueguitos IA")

BLANCO = (255, 255, 255)
NEGRO = (92, 29, 73)
ROJO = (228, 65, 110)
AZUL = (65, 136, 228)

# Variables principales del juego
jugador = None
bala_suelo = None
bala_aire = None
fondo = None
enemigo_tierra = None
menu = None

# Sistema de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True
subiendo = True

# Estados del juego
pausa = False
fuente_grande = pygame.font.SysFont('Arial', 28) 
fuente_media = pygame.font.SysFont('Arial', 24)
menu_activo = True 

# Modos de IA
modo_auto = False      # Red neuronal
modo_arbol = False     # Árbol de decisión
modo_knn = False       # K vecinos más cercanos

# Datos para entrenar IA
datos_modelo = []                    # Para entrenar saltos
modelo_entrenado = None              
modelo_entrenado_arbol = None        
movimiento_entrenado_arbol = None    
datos_movimiento = []                # Para entrenar movimiento
modelo_entrenado_movimiento = []     

intervalo_decidir_salto = 1
contador_decidir_salto = 0

# Cargar sprites
bala_img = pygame.transform.scale(cargar_imagen('assets/sprites/Teto/baguette.png'), (50, 50))
bala_aire_img = pygame.transform.scale(cargar_imagen('assets/sprites/Miku/puerro.png'), (30, 30))
fondo_img = cargar_imagen('assets/game/fondob.png')
enemigo_tierra_img = cargar_imagen('assets/sprites/Teto/teto.png')
enemigo_aereo_img = pygame.transform.scale(cargar_imagen('assets/sprites/Miku/Miku.png'), (120, 100))
menu_img = cargar_imagen('assets/game/menu.png')
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear hitboxes
jugador = pygame.Rect(50, h - 100, 32, 48)
bala_suelo = pygame.Rect(w - 50, h - 90, 16, 16)
bala_aire = pygame.Rect(0, -50, 16, 16)  
enemigo_tierra = pygame.Rect(w - 100, h - 130, 64, 64)
enemigo_aereo = pygame.Rect(0, 0, 64, 64)  

# Movimiento del enemigo aéreo
zigzag_direccion = 1  
zigzag_velocidad = 5  
enemigo_aereo_disparo_cooldown = 0
enemigo_aereo_disparo_intervalo = 60  # frames entre disparos
velocidad_bala_aire = [0, 5]  

menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)

# Animación
current_frame = 0
frame_speed = 10  
frame_count = 0

# Proyectiles
velocidad_bala_suelo = -10 
bala_disparada_suelo = False
bala_disparada_aire = False

# Fondo deslizante
fondo_x1 = 0
fondo_x2 = w
ultimo_disparo_aire = 0

# Cronómetro
tiempo_juego = 0
cronometro_pausado = False
inicio_tiempo = pygame.time.get_ticks()

# Funciones de movimiento
def mover_enemigo_aereo():
    """Mueve el enemigo aéreo en zigzag"""
    global enemigo_aereo, zigzag_direccion, enemigo_aereo_disparo_cooldown
    enemigo_aereo.x += zigzag_direccion * zigzag_velocidad
    enemigo_aereo_disparo_cooldown -= 1
    
    # Cambiar dirección en los bordes
    if enemigo_aereo.x <= 0 or enemigo_aereo.x >= 200 - enemigo_aereo.width:
        zigzag_direccion *= -1

def disparar_bala_aire():
    """Dispara proyectiles desde el enemigo aéreo"""
    global bala_aire, bala_disparada_aire, velocidad_bala_aire, ultimo_disparo_aire, enemigo_aereo_disparo_cooldown
    
    if not bala_disparada_aire and enemigo_aereo_disparo_cooldown <= 0 and 0 <= enemigo_aereo.x <= w:
        # Posicionar bala en el enemigo
        bala_aire.x = enemigo_aereo.x + enemigo_aereo.width // 2 - bala_aire.width // 2
        bala_aire.y = enemigo_aereo.y + enemigo_aereo.height
        
        velocidad_bala_aire[0] = 0  
        velocidad_bala_aire[1] = 5  
        
        bala_disparada_aire = True
        enemigo_aereo_disparo_cooldown = enemigo_aereo_disparo_intervalo
        ultimo_disparo_aire = pygame.time.get_ticks()

def disparar_bala():
    """Dispara proyectil terrestre con velocidad random"""
    global bala_disparada_suelo, velocidad_bala_suelo
    if not bala_disparada_suelo:
        velocidad_bala_suelo = random.randint(-8, -3)
        bala_disparada_suelo = True

def mover_jugador():
    """Controla el movimiento del jugador con las flechas"""
    global jugador, en_suelo, salto, pos_actual
    keys = pygame.key.get_pressed()
    pos_actual = 1  # Centro
    
    # Movimiento limitado a 0-200 píxeles
    if keys[pygame.K_LEFT] and jugador.x > 0:
        jugador.x -= 5
        pos_actual = 0
    if keys[pygame.K_RIGHT] and jugador.x < 200 - jugador.width:
        jugador.x += 5
        pos_actual = 2
    if keys[pygame.K_UP] and en_suelo:
        salto = True
        en_suelo = False
    
    # Calcular distancias para IA
    distancia_x = (jugador.centerx - bala_aire.centerx)
    distancia_y = (jugador.centery - bala_aire.centery)
    distancia_total = (distancia_x**2 + distancia_y**2) ** 0.5

# Cronómetro
def mostrar_cronometro():
    """Muestra el tiempo en pantalla"""
    minutos = tiempo_juego // 60000
    segundos = (tiempo_juego % 60000) // 1000
    texto = fuente_grande.render(f"Tiempo: {minutos:02d}:{segundos:02d}", True, BLANCO)
    pantalla.blit(texto, (10, 10))
    
def iniciar_cronometro():
    """Reinicia el cronómetro"""
    global inicio_tiempo, cronometro_activo, tiempo_juego
    inicio_tiempo = pygame.time.get_ticks()
    cronometro_activo = True
    tiempo_juego = 0

def actualizar_cronometro():
    """Actualiza el tiempo transcurrido"""
    global tiempo_juego, inicio_tiempo
    tiempo_juego = pygame.time.get_ticks() - inicio_tiempo

def reset_bala():
    """Resetear proyectil terrestre"""
    global bala_suelo, bala_disparada_suelo
    bala_suelo.x = w - 50
    bala_disparada_suelo = False
    
def reset_bala_aire():
    """Resetear proyectil aéreo"""
    global bala_aire, bala_disparada_aire
    bala_aire.y = -50
    bala_disparada_aire = False    

def manejar_salto():
    """Física del salto con gravedad"""
    global jugador, salto, salto_altura, gravedad, en_suelo, subiendo

    if salto:
        if subiendo:
            # Subiendo
            jugador.y -= salto_altura
            salto_altura -= gravedad
            if salto_altura <= 0:
                subiendo = False
        else:
            # Bajando
            jugador.y += salto_altura
            salto_altura += gravedad
            # Aterrizar
            if jugador.y >= h - 100:
                jugador.y = h - 100
                salto = False
                salto_altura = 15
                subiendo = True
                en_suelo = True

def update():
    """Actualiza todo el juego cada frame"""
    global bala_suelo, bala_aire, current_frame, frame_count, fondo_x1, fondo_x2
    mover_enemigo_aereo()
    
    # Sprites de Len (solo los necesarios para el ejemplo)
    jugador_rect = [
        pygame.transform.scale(pygame.image.load(r'assets/sprites/Len/lenQ.png'), (44, 55)),
        pygame.transform.scale(pygame.image.load(r'assets/sprites/Len/len0.png'), (44, 55)),
        # ... (otros frames de animación)
    ]

    jugador_rect_salto = [
        pygame.transform.scale(pygame.image.load(r'assets/sprites/Len/len8.png'), (44, 55)),
        pygame.transform.scale(pygame.image.load(r'assets/sprites/Len/len9.png'), (44, 55)),
    ]

    # Mover fondo (efecto parallax)
    fondo_x1 -= 3
    fondo_x2 -= 3
    if fondo_x1 <= -w:
        fondo_x1 = w
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar fondo
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    if salto:
        if subiendo:
            pantalla.blit(jugador_rect_salto[0], (jugador.x, jugador.y))
        else:
            pantalla.blit(jugador_rect_salto[1], (jugador.x, jugador.y))
    else:
        frame_count += 10
        if frame_count >= frame_speed:
            current_frame = (current_frame + 1) % len(jugador_rect)
            frame_count = 0
        pantalla.blit(jugador_rect[current_frame], (jugador.x, jugador.y))

    # Dibujar enemigos
    pantalla.blit(enemigo_tierra_img, (enemigo_tierra.x, enemigo_tierra.y))
    pantalla.blit(enemigo_aereo_img, (enemigo_aereo.x, enemigo_aereo.y+75))

    # Mover y dibujar proyectiles
    if bala_disparada_suelo:
        bala_suelo.x += velocidad_bala_suelo
        pantalla.blit(bala_img, (bala_suelo.x, bala_suelo.y))
        
    if bala_disparada_aire:
        bala_aire.x += velocidad_bala_aire[0]  
        bala_aire.y += velocidad_bala_aire[1]  
        pantalla.blit(bala_aire_img, (bala_aire.x, bala_aire.y))

    # Resetear balas que salen de pantalla
    if bala_suelo.x < 0:
        reset_bala()
    if bala_aire.y > h or bala_aire.x < 0 or bala_aire.x > w:
        reset_bala_aire()

    # Detectar colisiones - Game Over
    if jugador.colliderect(bala_suelo) or  jugador.colliderect(bala_aire): 
        print("¡Te mataron!")
        reiniciar_juego()

def guardar_datos():
    """Guarda datos del juego para entrenar la IA"""
    global jugador, bala_suelo, velocidad_bala_suelo, salto
    
    # Datos para modelo de salto
    distancia_suelo = abs(jugador.x - bala_suelo.x)
    salto_hecho = 1 if salto else 0
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)
    hay_bala_aire = 1 if bala_disparada_aire else 0

    datos_modelo.append((
        velocidad_bala_suelo,
        distancia_suelo,
        distancia_aire_x,
        distancia_aire_y,
        hay_bala_aire,
        jugador.x,
        salto_hecho
    ))

    # Datos para modelo de movimiento
    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)
    datos_movimiento.append((
        jugador.x,
        jugador.y,
        bala_aire.centerx,
        bala_aire.centery,
        bala_suelo.x,
        bala_suelo.y,
        distancia_bala_suelo,
        1 if salto else 0,
        pos_actual
    ))

def pausa_juego():
    """Pausar/despausar el juego"""
    global pausa, cronometro_pausado, inicio_tiempo
    pausa = not pausa
    cronometro_pausado = pausa
    if pausa:
        imprimir_datos()
    else:
        inicio_tiempo = pygame.time.get_ticks() - (tiempo_juego * 1000)
        print("Juego reanudado.")

def mostrar_menu():
    """Menú principal con todas las opciones"""
    global menu_activo, modo_auto, modo_arbol, modo_knn
    global datos_modelo, datos_movimiento
    global modelo_entrenado, modelo_entrenado_movimiento
    global modelo_entrenado_arbol, movimiento_entrenado_arbol

    pantalla.fill(NEGRO)
    actualizar_cronometro()
    mostrar_cronometro()
    
    # Opciones del menú
    pantalla.blit(fuente_grande.render("MENU PRINCIPAL", True, BLANCO), (w//2 - 120, h//4 - 30))
    pantalla.blit(fuente_media.render("Presiona 'R' para Modo Red Neuronal", True, BLANCO), (w//2 - 180, h//4 + 30))
    pantalla.blit(fuente_media.render("Presiona 'M' para Modo Manual", True, BLANCO), (w//2 - 180, h//4 + 55))
    pantalla.blit(fuente_media.render("Presiona 'E' para Entrenar Modelo", True, BLANCO), (w//2 - 180, h//4 + 80))
    pantalla.blit(fuente_media.render("Presiona 'A' para Modo Arbol", True, BLANCO), (w//2 - 180, h//4 + 105))
    pantalla.blit(fuente_media.render("Presiona 'K' para KNN", True, BLANCO), (w//2 - 180, h//4 + 130))
    pantalla.blit(fuente_media.render("Presiona 'Q' para Salir", True, BLANCO), (w//2 - 180, h//4 + 155))
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                # Red neuronal
                if evento.key == pygame.K_r:
                    if(datos_modelo and datos_movimiento):
                        modo_auto = True
                        modo_arbol = False
                        modo_knn = False
                        menu_activo = False
                        iniciar_cronometro()
                    else:
                        print("Necesitas datos! Juega primero con 'M'")
                
                # Entrenar modelos
                if evento.key == pygame.K_e:
                    if len(datos_modelo) > 0 and len(datos_movimiento) > 0:
                        modelo_entrenado = entrenar_modelo(datos_modelo)
                        modelo_entrenado_movimiento = entrenar_red_movimiento(datos_movimiento)

                # Modo manual (para generar datos)
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    modo_arbol = False
                    modo_knn = False
                    menu_activo = False
                    datos_modelo = []
                    datos_movimiento = []
                    iniciar_cronometro()
                
                # Árbol de decisión
                elif evento.key == pygame.K_a:
                    modo_auto = False
                    modo_arbol = True
                    modo_knn = False
                    menu_activo = False
                    modelo_entrenado_arbol = entrenar_arbol_salto(datos_modelo)
                    movimiento_entrenado_arbol = entrenar_arbol_movimiento(datos_movimiento)
                    iniciar_cronometro()
                
                elif evento.key == pygame.K_x:
                    reiniciar_juego()    
                
                # KNN
                if evento.key == pygame.K_k:
                    modo_auto = False
                    modo_arbol = False
                    modo_knn = True
                    menu_activo = False
                    modelo_entrenado = entrenar_knn_salto(datos_modelo)
                    modelo_entrenado_movimiento = entrenar_knn_movimiento(datos_movimiento)
                    iniciar_cronometro()    
                
                # Salir
                elif evento.key == pygame.K_q:
                    imprimir_datos()
                    pygame.quit()
                    exit()

def reiniciar_juego():
    """Resetea todo y vuelve al menú"""
    global menu_activo, jugador, bala_suelo, bala_aire, enemigo_tierra, bala_disparada_suelo, bala_disparada_aire, salto, en_suelo
    menu_activo = True
    jugador.x, jugador.y = 50, h - 100
    bala_suelo.x = w - 50
    bala_aire.y = -50
    enemigo_tierra.x, enemigo_tierra.y = w - 100, h - 100
    bala_disparada_suelo = False
    bala_disparada_aire = False
    salto = False
    en_suelo = True
    imprimir_datos()
    mostrar_menu()
    
def imprimir_datos():
    """Debug: imprime los datos recolectados"""
    for dato in datos_movimiento:
        print(dato)
        
def main():
    """Loop principal del juego"""
    global salto, en_suelo, bala_disparada_suelo, bala_disparada_aire, contador_decidir_salto

    reloj = pygame.time.Clock()
    mostrar_menu()
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_q:
                    imprimir_datos()
                    pygame.quit()
                    exit()

        if not pausa:
            # Modo manual - el jugador controla
            if not modo_auto and not modo_arbol and not modo_knn:
                mover_jugador()
                if salto:
                    manejar_salto()
                guardar_datos()

            # Modo red neuronal - IA controla
            if modo_auto and modelo_entrenado and modelo_entrenado_movimiento:
                    salto, en_suelo = decidir_salto(jugador, bala_suelo, velocidad_bala_suelo, bala_aire, bala_disparada_aire, modelo_entrenado, salto, en_suelo)
                    manejar_salto()
                    jugador.x, pos_actual = decidir_movimiento(jugador, bala_aire, modelo_entrenado_movimiento, salto, bala_suelo)
                    mover_jugador()
            
            # Modo KNN - IA con vecinos cercanos
            if modo_knn:         
                    salto, en_suelo = decidir_salto_knn(jugador, bala_suelo, velocidad_bala_suelo, bala_aire, bala_disparada_aire, modelo_entrenado, salto, en_suelo)
                    manejar_salto()
                    jugador.x, pos_actual = decidir_movimiento_knn(jugador, bala_aire, modelo_entrenado_movimiento, salto, bala_suelo)

            # Modo árbol de decisión - IA con reglas
            if modo_arbol:         
                    salto, en_suelo = decidir_salto_arbol(jugador, bala_suelo, velocidad_bala_suelo, bala_aire, bala_disparada_aire, modelo_entrenado_arbol, salto, en_suelo)
                    manejar_salto()
                    jugador.x, pos_actual = decidir_movimiento_arbol(jugador, bala_aire, movimiento_entrenado_arbol, salto, bala_suelo)

            # Disparar balas
            if not bala_disparada_suelo:
                disparar_bala()
            disparar_bala_aire()
            
            # Actualizar todo
            update()
            mostrar_cronometro()
            if not pausa:
                 if not cronometro_pausado:
                    actualizar_cronometro()
        
        pygame.display.flip()
        reloj.tick(30)  # 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()