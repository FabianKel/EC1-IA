import random
import matplotlib.pyplot as plt
from collections import deque

def generate_random_map(size=4, max_intentos=100, min_holes=4, max_holes=4):
    """
    Genera un mapa aleatorio con control de agujeros
    :param min_holes: Mínimo de agujeros (H) en el mapa (sin contar S/G)
    :param max_holes: Máximo de agujeros (H) en el mapa (sin contar S/G)
    """
    def es_camino_valido(mapa):
        n = len(mapa)
        visitado = set()
        cola = deque([(0, 0)])

        while cola:
            x, y = cola.popleft()
            if (x, y) == (n-1, n-1):
                return True
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0<=nx<n and 0<=ny<n and mapa[nx][ny] != 'H' and (nx,ny) not in visitado:
                    visitado.add((nx, ny))
                    cola.append((nx, ny))
        return False

    for _ in range(max_intentos):
        # Generar mapa base
        mapa = [['F' for _ in range(size)] for _ in range(size)]
        mapa[0][0] = 'S'
        mapa[-1][-1] = 'G'

        # Calcular posiciones válidas para agujeros (excluyendo S y G)
        posiciones_validas = [(i,j) for i in range(size) for j in range(size) 
                            if (i,j) not in [(0,0), (size-1, size-1)]]

        # Generar agujeros aleatorios dentro del rango
        num_holes = random.randint(min_holes, max_holes)
        holes = random.sample(posiciones_validas, num_holes)
        
        for i, j in holes:
            mapa[i][j] = 'H'

        # Validar y retornar si es válido
        mapa_str = [''.join(row) for row in mapa]
        if es_camino_valido(mapa_str):
            return mapa_str

    raise ValueError(f"No se generó mapa válido en {max_intentos} intentos")



def plot_map(frozen_map, filename='generated_map.png'):
    plt.figure(figsize=(6, 6))
    
    # Convertir el mapa a una matriz numérica
    map_matrix = []
    color_mapping = {'S': 0, 'F': 1, 'H': 2, 'G': 3}
    for row in frozen_map:
        map_matrix.append([color_mapping[cell] for cell in row])
    
    # Crear mapa de colores personalizado
    cmap = plt.cm.colors.ListedColormap([
        '#2ecc71',   # S (Verde)
        '#3498db',   # F (Azul claro)
        '#2c3e50',   # H (Azul oscuro)
        '#f1c40f'    # G (Amarillo/Oro)
    ])
    
    plt.imshow(map_matrix, cmap=cmap, interpolation='nearest')
    
    # Añadir anotaciones y leyenda
    plt.title('Mapa Generado', fontsize=14, pad=20)
    plt.xticks([])
    plt.yticks([])
    
    # Crear leyenda personalizada
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Inicio (S)',
                  markerfacecolor='#2ecc71', markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Hielo (F)',
                  markerfacecolor='#3498db', markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Agujero (H)',
                  markerfacecolor='#2c3e50', markersize=12),
        plt.Line2D([0], [0], marker='s', color='w', label='Meta (G)',
                  markerfacecolor='#f1c40f', markersize=12)
    ]
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
             loc='upper left', borderaxespad=0.)
    
    
    plt.text(0, 0, '→', ha='center', va='center', 
            fontsize=24, color='white')
    plt.text(len(frozen_map)-1, len(frozen_map[-1])-1, '★', 
            ha='center', va='center', 
            fontsize=24, color='white')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Mapa generado guardado como {filename}")