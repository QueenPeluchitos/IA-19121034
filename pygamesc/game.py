import pygame
import random
import tensorflow as tf
import numpy as np
import pickle
import os # Para verificar si el archivo del modelo existe

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Coloritos
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
VERDE_TEXTO = (100, 255, 100)
ROJO_TEXTO = (255, 100, 100)

# Variables del jugador, bala, nave, fondo y menú
jugador_rect = None 
bala_rect = None
fondo = None
nave_rect = None
menu = None

# Variables de salto
salto = False
salto_altura_inicial = 15  # Velocidad inicial de salto
salto_altura_actual = salto_altura_inicial
gravedad = 1
en_suelo = True

# Variables de pausa y menú
pausa = False
fuente_grande = pygame.font.SysFont('Arial', 28)
fuente_media = pygame.font.SysFont('Arial', 24)
fuente_pequena = pygame.font.SysFont('Arial', 18)
menu_activo = True
modo_auto = False  # Indica si el modo de juego es automático

# Lista para guardar los datos de velocidad, distancia y salto
datos_modelo = []

# Carga las imágenes
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
    print("Asegúrate de que la carpeta 'assets' esté en el mismo directorio que el script y contenga las imágenes.")
    pygame.quit()
    exit()


fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear el rectángulo del jugador y de la bala
jugador_rect = pygame.Rect(50, h - 100, 32, 48) # x, y, ancho, alto
bala_rect = pygame.Rect(w - 50, h - 90, 16, 16)
nave_rect = pygame.Rect(w - 100, h - 100, 64, 64)

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para la bala
velocidad_bala_actual = -10  # Velocidad de la bala hacia la izquierda
bala_disparada = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

# Cargar el modelo entrenado (modo automático)
modelo = None 
MODELO_PATH = 'modelo_salto.keras'
try:
    if os.path.exists(MODELO_PATH):
        modelo = tf.keras.models.load_model(MODELO_PATH)
        print(f"Modelo '{MODELO_PATH}' cargado exitosamente.")
    else:
        print(f"Advertencia: Archivo de modelo '{MODELO_PATH}' no encontrado. El modo automático no funcionará hasta que se entrene un modelo.")
except Exception as e: # Captura excepciones más generales de Keras
    print(f"Advertencia: No se pudo cargar '{MODELO_PATH}'. El modo automático no funcionará. Error: {e}")


# Función para decidir si saltar usando el modelo
def decision_auto():
    global jugador_rect, bala_rect, salto, en_suelo, velocidad_bala_actual, modelo

    if not modelo:
        # print("Modo automático no disponible: modelo no cargado o no válido.") 
        return

    if not en_suelo:
        return

    # Asegurar que sea float
    velocidad = float(velocidad_bala_actual) 
    distancia = float(abs(jugador_rect.x - bala_rect.x)) 

    # El modelo espera una entrada de la forma
    entrada = np.array([[velocidad, distancia]], dtype=np.float32)
    try:
        prediccion = modelo.predict(entrada, verbose=0)[0][0]

        if prediccion > 0.5:  # Umbral de decisión
            salto = True
            en_suelo = False
    except Exception as e:
        print(f"Error durante la predicción del modelo: {e}")


# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala_actual
    if not bala_disparada:
        velocidad_bala_actual = random.randint(-8, -3)  # Velocidad aleatoria negativa para la bala
        bala_disparada = True

# Función para reiniciar la posición de la bala
def reset_bala():
    global bala_rect, bala_disparada
    bala_rect.x = w - 50  # Reiniciar la posición de la bala
    bala_disparada = False

# Función para manejar el salto
def manejar_salto():
    global jugador_rect, salto, salto_altura_actual, gravedad, en_suelo, salto_altura_inicial

    if salto:
        jugador_rect.y -= salto_altura_actual  # Mover al jugador hacia arriba
        salto_altura_actual -= gravedad  # Aplicar gravedad

        # Si el jugador llega al suelo, detener el salto
        if jugador_rect.y >= h - 100:
            jugador_rect.y = h - 100
            salto = False
            salto_altura_actual = salto_altura_inicial  # Restablecer la velocidad de salto
            en_suelo = True
    elif not en_suelo and jugador_rect.y < h - 100 : # Si no está saltando pero está en el aire
        jugador_rect.y -= salto_altura_actual # Sigue aplicando el movimiento vertical
        salto_altura_actual -= gravedad
        if jugador_rect.y >= h - 100:
            jugador_rect.y = h - 100
            salto_altura_actual = salto_altura_inicial
            en_suelo = True


# Función para actualizar el juego
def update_game_state():
    global bala_rect, velocidad_bala_actual, current_frame, frame_count, fondo_x1, fondo_x2

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    if fondo_x1 <= -w:
        fondo_x1 = w
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar los fondos
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    pantalla.blit(jugador_frames[current_frame], (jugador_rect.x, jugador_rect.y))

    # Dibujar la nave
    pantalla.blit(nave_img, (nave_rect.x, nave_rect.y))

    # Mover y dibujar la bala
    if bala_disparada:
        bala_rect.x += velocidad_bala_actual

    if bala_rect.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala_rect.x, bala_rect.y))

    # Colisión entre la bala y el jugador
    if jugador_rect.colliderect(bala_rect):
        print("Colisión detectada!")
        reiniciar_juego_a_menu()


# Función para guardar datos del modelo en modo manual
def guardar_datos_para_modelo():
    global jugador_rect, bala_rect, velocidad_bala_actual, salto, datos_modelo
    distancia = abs(jugador_rect.x - bala_rect.x)
    # Captura si un salto fue INICIADO en este frame para evitar la bala
    salto_decision = 1 if salto else 0
    datos_modelo.append((float(velocidad_bala_actual), float(distancia), int(salto_decision)))


# Función para pausar el juego
def toggle_pausa():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado.")
    else:
        print("Juego reanudado.")

def guardar_datos_a_archivo():
    global datos_modelo
    if datos_modelo:
        try:
            with open('datos_entrenamiento.pkl', 'wb') as f:
                pickle.dump(datos_modelo, f)
            print(f"Datos de entrenamiento ({len(datos_modelo)} puntos) guardados en 'datos_entrenamiento.pkl'")
        except Exception as e:
            print(f"Error al guardar datos: {e}")
    else:
        print("No hay nuevos datos para guardar.")

# Menú
def mostrar_menu():
    global menu_activo, modo_auto, modelo
    menu_activo = True

    while menu_activo:
        pantalla.fill(NEGRO) # Limpiar pantalla para el menú
        
        texto_titulo = fuente_grande.render("MENU PRINCIPAL", True, BLANCO)
        pantalla.blit(texto_titulo, (w // 2 - texto_titulo.get_width() // 2, h // 4))

        opcion_auto = fuente_media.render("Presiona 'A' para Modo Automático", True, BLANCO)
        opcion_manual = fuente_media.render("Presiona 'M' para Modo Manual (Recolectar Datos)", True, BLANCO)
        opcion_entrenar = fuente_media.render("Presiona 'T' para Entrenar Modelo", True, BLANCO)
        opcion_salir = fuente_media.render("Presiona 'Q' para Salir del Juego", True, BLANCO)

        pantalla.blit(opcion_auto, (w // 2 - opcion_auto.get_width() // 2, h // 2 - 60))
        pantalla.blit(opcion_manual, (w // 2 - opcion_manual.get_width() // 2, h // 2 - 20))
        pantalla.blit(opcion_entrenar, (w // 2 - opcion_entrenar.get_width() // 2, h // 2 + 20))
        pantalla.blit(opcion_salir, (w // 2 - opcion_salir.get_width() // 2, h // 2 + 60))

        # Mostrar estado del modelo
        if modelo:
            estado_modelo_txt = fuente_pequena.render("Modelo cargado. Modo Auto disponible.", True, VERDE_TEXTO)
        else:
            estado_modelo_txt = fuente_pequena.render("Modelo no cargado. Entrena o carga uno para Modo Auto.", True, ROJO_TEXTO)
        pantalla.blit(estado_modelo_txt, (w // 2 - estado_modelo_txt.get_width() // 2, h // 2 + 100))


        pygame.display.flip()

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                guardar_datos_a_archivo()
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    if modelo:
                        modo_auto = True
                        menu_activo = False
                    else:
                        print("No se puede iniciar Modo Automático: El modelo no está cargado.")
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_t:
                    print("Iniciando entrenamiento del modelo...")
                    entrenar_modelo_desde_datos()
                    # Recargar el modelo después de entrenar
                    try:
                        if os.path.exists(MODELO_PATH):
                            modelo = tf.keras.models.load_model(MODELO_PATH)
                            print(f"Modelo '{MODELO_PATH}' recargado exitosamente después del entrenamiento.")
                        else:
                            print(f"Archivo de modelo '{MODELO_PATH}' no encontrado después del entrenamiento.")
                            modelo = None 
                    except Exception as e:
                        print(f"Error al recargar modelo después del entrenamiento: {e}")
                        modelo = None
                elif evento.key == pygame.K_q:
                    guardar_datos_a_archivo()
                    pygame.quit()
                    exit()

# Función para reiniciar el juego tras la colisión y volver al menú
def reiniciar_juego_a_menu():
    global menu_activo, jugador_rect, bala_rect, nave_rect, bala_disparada, salto, en_suelo, salto_altura_actual, salto_altura_inicial
    
    print("Volviendo al menú...")

    menu_activo = True
    jugador_rect.x, jugador_rect.y = 50, h - 100
    bala_rect.x = w - 50
    nave_rect.x, nave_rect.y = w - 100, h - 100
    bala_disparada = False
    salto = False
    en_suelo = True
    salto_altura_actual = salto_altura_inicial
    # mostrar menu

    # Aquí empieza el entrenamiento del modelo
def entrenar_modelo_desde_datos():
    global datos_modelo, modelo # Para actualizar la variable global del modelo
    
    datos_para_entrenamiento = []
    # Usar datos de la sesión actual

    # Cargar todos los datos guardados previamente
    try:
        with open('datos_entrenamiento.pkl', 'rb') as f:
            datos_cargados_archivo = pickle.load(f)
            datos_para_entrenamiento.extend(datos_cargados_archivo)
        print(f"Cargados {len(datos_cargados_archivo)} puntos de datos desde 'datos_entrenamiento.pkl'")
    except FileNotFoundError:
        print("Archivo 'datos_entrenamiento.pkl' no encontrado. Solo se usarán datos de la sesión actual (si los hay).")
    except Exception as e:
        print(f"Error al cargar datos desde archivo: {e}")

    # Añadir datos de la sesión actual si no están ya guardados
    if datos_modelo not in datos_para_entrenamiento: # Evita duplicar la lista entera si ya está
         datos_para_entrenamiento.extend(datos_modelo)


    if not datos_para_entrenamiento:
        print("No hay datos para entrenar (ni en sesión actual ni en archivo). Juega en modo manual para recolectar.")
        return

    print(f"Entrenando con un total de {len(datos_para_entrenamiento)} puntos de datos.")

    # Preparar los datos para TensorFlow
    entradas = np.array([[dato[0], dato[1]] for dato in datos_para_entrenamiento], dtype=np.float32)
    salidas = np.array([dato[2] for dato in datos_para_entrenamiento], dtype=np.float32).reshape(-1, 1) 
    
    # Definir el modelo
    nuevo_modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)), # Capa de entrada
        tf.keras.layers.Dense(16, activation='relu'),                  # Capa oculta
        tf.keras.layers.Dense(1, activation='sigmoid')                # Capa de salida
    ])
    
    nuevo_modelo.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
    
    print("Entrenando el modelo...")
    try:
        nuevo_modelo.fit(entradas, salidas, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
        
        nuevo_modelo.save(MODELO_PATH)
        print(f"Modelo entrenado y guardado como '{MODELO_PATH}'")
        modelo = nuevo_modelo # Actualizar la variable global del modelo en el juego
    except Exception as e:
        print(f"Error durante el entrenamiento o guardado del modelo: {e}")


def main():
    global salto, en_suelo, bala_disparada, pausa, menu_activo, jugador_rect, salto_altura_actual, salto_altura_inicial

    reloj = pygame.time.Clock()
    
    correr = True
    while correr:
        if menu_activo:
            mostrar_menu() # Esto bloqueará hasta que se seleccione una opción

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if not menu_activo and not pausa: # Controles de juego solo si no está en menú o pausa
                    if evento.key == pygame.K_SPACE and en_suelo:
                        salto = True
                        en_suelo = False
                        salto_altura_actual = salto_altura_inicial # Reiniciar altura de salto al inicio del salto
                
                # Controles globales
                if evento.key == pygame.K_p:
                    if not menu_activo : # No pausar si estamos en el menú
                        toggle_pausa()
                if evento.key == pygame.K_q: # Salir desde cualquier punto
                    correr = False
                if evento.key == pygame.K_m and pausa: # Volver al menu si está en pausa y presiona M
                     reiniciar_juego_a_menu()


        if not menu_activo: # Solo ejecutar lógica de juego si no estamos en el menú
            if not pausa:
                # Modo automático
                if modo_auto:
                    decision_auto() 
                    # manejar_salto se llama independientemente de si es auto o manual si salto es True

                # Siempre manejar salto si la variable 'salto' es True o si está cayendo
                if salto or (not en_suelo and jugador_rect.y < h - 100):
                    manejar_salto()

                # Guardar los datos si estamos en modo manual y el juego está corriendo
                if not modo_auto:
                    guardar_datos_para_modelo()

                # Actualizar el juego
                if not bala_disparada:
                    disparar_bala()
                
                update_game_state() # Esto dibuja todo

            else: # Si el juego está en pausa
                pantalla.fill(NEGRO) # Fondo oscuro para la pausa
                texto_pausa = fuente_grande.render("PAUSA", True, BLANCO)
                texto_instr_pausa = fuente_pequena.render("Presiona 'P' para reanudar, 'M' para Menú", True, BLANCO)
                pantalla.blit(texto_pausa, (w // 2 - texto_pausa.get_width() // 2, h // 2 - 30))
                pantalla.blit(texto_instr_pausa, (w // 2 - texto_instr_pausa.get_width() // 2, h // 2 + 10))


        pygame.display.flip()
        reloj.tick(30)  # Limitar el juego a 30 FPS

    guardar_datos_a_archivo() # Guardar datos al salir del bucle principal
    pygame.quit()

if __name__ == "__main__":
    main()