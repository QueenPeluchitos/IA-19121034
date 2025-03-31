import pygame
from queue import PriorityQueue

# Inicio de las fuentes
pygame.font.init()

# Ventana 
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Algoritomo asterisco, Entregable 1")

#Coloritos
BLANCO = (245, 239, 230)
NEGRO = (36, 34, 32)
GRIS = (141, 138, 134)
VERDE = (137, 205, 139)
ROJO = (223, 30, 76)
NARANJA = (246, 129, 59)
PURPURA = (175, 112, 207)  
AZUL_CIELO = (145, 182, 229)

class Nodo:
    def __init__(self, fila, col, ancho, total_filas): #constructor
        #Representan a las celdas
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO #Le da colores a las celdas
        self.ancho = ancho
        self.total_filas = total_filas
        self.fuente = pygame.font.SysFont("Arial", 15) #Fuente y tamaño
        #Valores para la costo g, heuristica h, funcion f
        self.g = float("inf") #Inicio a fin del camino
        self.h = float("inf") #Estimacion de la distancia
        self.f = float("inf") #Costo total
        #Valores para el nodo padre y contador
        self.padre = None
        self.contador = 0
        
    #Restablece el nodo a su estado inicial
    def get_pos(self):
        return self.fila, self.col
   
    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    #Estados de paredes, inicio y fin
    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA
    #camino cerrado, abierto y camino óptimo
    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_abierto(self):
        self.color = AZUL_CIELO

    def hacer_camino(self):
        self.color = VERDE
    #Actualiza los costos de los movimientos
    def actualizar_costos(self, g, h, padre, contador):
        self.g = g
        self.h = h
        self.f = g + h
        self.padre = padre
        self.contador = contador
    #Ventana, numero de la celda y coordenas de cada celda
    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))
        self.dibujar_numero(ventana)

    def dibujar_numero(self, ventana):
        numero = self.col * self.total_filas + self.fila + 1
        texto = self.fuente.render(str(numero), True, NEGRO)
        ventana.blit(texto, (self.x + 5, self.y + 5))

    def __lt__(self, otro):
        return self.f < otro.f

def heuristica(pos1, pos2): #Distancia Manhattan entre 2 puntos
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)

def vecinos(nodo, grid):
    filas = nodo.total_filas
    vecinos_list = []
    fila, col = nodo.fila, nodo.col
    
    #valores de los movimientos
    movimientos = [
        (-1, 0, 10),  (1, 0, 10), (0, -1, 10), (0, 1, 10), # izquierda, derecha, abajo, arriba
        (-1, -1, 14), (-1, 1, 14), (1, -1, 14), (1, 1, 14) # diagonales de izquierda abajo, izquierda arriba, derecha abajo, derecha arriba
    ]
    for df, dc, costo in movimientos: 
        nueva_fila, nueva_col = fila + df, col + dc
        if 0 <= nueva_fila < filas and 0 <= nueva_col < filas:
            vecino = grid[nueva_fila][nueva_col]
            vecinos_list.append((vecino, costo))
    return vecinos_list

def crear_grid(filas, ancho): #Crea la cuadrícula de nodos
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho): #Dibuja la cuadrícula
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho): #Dibuja la ventana y los nodos
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def reconstruir_camino(nodo_actual, ventana, grid, filas, ancho): #Reconstruye el camino óptimo
    while nodo_actual.padre:
        nodo_actual.hacer_camino()
        nodo_actual = nodo_actual.padre
        dibujar(ventana, grid, filas, ancho)

def algoritmo_heuristico(grid, inicio, fin, ventana, ancho): #Algoritmo A*
    open_set = PriorityQueue()
    open_set.put((0, inicio))
    inicio.actualizar_costos(0, heuristica(inicio.get_pos(), fin.get_pos()), None, 1)
    visitados = {inicio}
    contador = 2

    while not open_set.empty(): ## Mientras haya nodos abiertos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        pygame.time.delay(50) # Controla el delay

        nodo_actual = open_set.get()[1] # Extrae el nodo con menor costo f

        if nodo_actual == fin: # Si se llega al nodo final
            reconstruir_camino(fin, ventana, grid, len(grid), ancho)
            return True

        for vecino, costo in vecinos(nodo_actual, grid): # Obtiene los vecinos del nodo actual
            if vecino.color == NEGRO:
                continue

            nuevo_g = nodo_actual.g + costo 
            if nuevo_g < vecino.g:
                vecino.actualizar_costos(nuevo_g, heuristica(vecino.get_pos(), fin.get_pos()), nodo_actual, contador)
                contador += 1
                vecino.hacer_abierto()
                open_set.put((vecino.f, vecino))

        if nodo_actual != inicio: 
            nodo_actual.hacer_cerrado()

        dibujar(ventana, grid, len(grid), ancho) 

    return False 

def obtener_click_pos(pos, filas, ancho): 
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def main(ventana, ancho): 
    FILAS = 10
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True 
    while corriendo:
        dibujar(ventana, grid, FILAS, ancho) 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_CTRL: 
                    for fila in grid:
                        for nodo in fila:
                            nodo.restablecer()
                    inicio = None
                    fin = None

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    algoritmo_heuristico(grid, inicio, fin, ventana, ancho)

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)