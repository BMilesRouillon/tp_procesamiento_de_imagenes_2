"""
problema_2.py
=============
Localización y segmentación de los 6 caracteres de patentes argentinas (formato
viejo: 3 letras + 3 números, caja negra con caracteres claros) en fotos de 640x360.

El flujo completo tiene 5 etapas (cada una es una sección de este archivo):

  1) LOCALIZAR la patente en la foto y recortarla.
        detectar_punto()  ->  recortar()
  2) BINARIZAR el recorte (blanco/negro) y buscar CANDIDATOS a caracteres
     con componentes conexas.
        preparar_foto()   ->  componentes()
  3) ELEGIR los caracteres, igualar sus cajas y, si faltan, DEDUCIR dónde están
     usando la geometría de la patente (alineación, separación constante,
     mitad izquierda/derecha).
        cajas_de_area_similar()  ->  igualar_cajas()  ->  proyectar_lado()
  4) ENDEREZAR: con las cajas ya definidas, se mide la inclinación de la patente (recta por los
     centros de la primera y la última caja), se GIRA el recorte y, con la MISMA matriz, se giran
     las cajas. Los caracteres se recortan de la imagen girada (fila horizontal).
        calcular_giro()  ->  girar()
     La detección de las etapas 2 y 3 se hace SIN rotar: el giro no puede empeorarla.
  5) SEGMENTAR LA PLACA: en el recorte ya derecho, se parte de la fila de caracteres y se
     avanza hacia afuera (izquierda, derecha, arriba y abajo) mientras la columna o fila
     siga siendo mayormente oscura. Así se obtiene la caja negra de la patente.
        segmentar_placa()

Punto de entrada:   analizar_foto(ruta)   -> Resultado
Para verlo:         python problema_2.py   (abre una ventana/pestaña por foto e informa por consola)
                    ver_resultados(rutas)  (desde un notebook o un script)

Convenciones
------------
* Las imágenes son matrices NumPy (alto x ancho x 3, orden BGR de OpenCV).
* Coordenadas: x crece hacia la derecha, y crece hacia ABAJO (como en OpenCV).
* Una "caja" es el rectángulo mínimo que rodea a un carácter (ver `Caja`).

Este módulo UBICA los caracteres; todavía no reconoce cuál es cada uno.

Requisitos: opencv-python, numpy, matplotlib.
"""

import sys
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PARÁMETROS (todos en un solo lugar; se pueden ajustar)
# ============================================================

# --- Etapa 1: localización ---
MARGEN_IZQ, MARGEN_DER, MARGEN_SUP = 0.20, 0.20, 0.20   # zona de la foto donde NUNCA hay patentes (fracción)
V_NEGRO = 80          # un píxel es "negro" si su brillo (canal V de HSV) es menor a esto (0-255)
ALTO_MIN_TRAZO = 3    # un borde vertical debe medir al menos esto (px) para no ser ruido
VENTANA_W, VENTANA_H = 0.11, 0.09   # tamaño de la ventana de búsqueda (fracción de la foto)
RECORTE_W, RECORTE_H = 0.16, 0.14   # tamaño del recorte final (fracción de la foto)

# --- Etapa 2: binarización y candidatos ---
UMBRAL_BINARIO = 127  # tras estirar el contraste: > umbral -> blanco
AREA_MIN = 6          # una componente con menos píxeles que esto es ruido
ALTO_REL = (0.20, 0.70)   # un carácter mide entre 20% y 70% del alto del recorte
CONECTIVIDAD = 4      # 4: solo vecinos arriba/abajo/izq/der; 8: también diagonales

# --- Etapa 3: elección y proyección ---
TOL_AREA = 0.25       # dos cajas son "parecidas" si sus áreas difieren menos de 25% (relativo)
N_CARACTERES = 6
MARGEN_PROYECCION = 1.5   # ver proyectar_lado()

# --- Etapa 4: enderezado ---
ROTAR = True                              # False: no girar (el Resultado queda sin imagen girada)
GIRO_INTERPOLACION = cv2.INTER_CUBIC      # interpolación al girar la imagen (no afecta a la detección)
GIRO_BORDE = cv2.BORDER_REPLICATE         # relleno de los bordes que quedan vacíos al girar

# --- Etapa 5: segmentación de la placa ---
FRAC_OSCURO = 0.5     # una columna/fila sigue siendo placa si más de la mitad de sus píxeles son oscuros


# ============================================================
# ESTRUCTURAS DE DATOS
# ============================================================

class Caja(NamedTuple):
    """Rectángulo que rodea a un carácter: esquina superior izquierda (x, y), ancho w y alto h."""
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        """Coordenada x del centro de la caja (no del centro de masa del carácter)."""
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        """Coordenada y del centro de la caja."""
        return self.y + self.h / 2

    @property
    def area(self) -> int:
        """Área de la CAJA (ancho x alto), no de los píxeles del carácter."""
        return self.w * self.h


class Proyeccion(NamedTuple):
    """Caja deducida (no detectada) de un carácter que falta. Coordenadas decimales."""
    lado: str      # "izq" o "der": en qué mitad de la patente está
    motivo: str    # "en el medio", "hacia el centro" o "en el extremo"
    x0: float      # esquina superior izquierda
    y0: float
    x1: float      # esquina inferior derecha
    y1: float


@dataclass
class Resultado:
    """Todo lo que produce analizar_foto() para una foto."""
    recorte: np.ndarray                 # recorte color de la patente
    estirado: np.ndarray                # recorte en grises con contraste estirado (0-255)
    binaria: np.ndarray                 # recorte blanco/negro (0 o 255)
    candidatas: List[Caja] = field(default_factory=list)   # todas las componentes que podrían ser caracteres
    originales: List[Caja] = field(default_factory=list)   # las elegidas por área parecida, tal como se detectaron
    encontradas: List[Caja] = field(default_factory=list)  # las mismas, igualadas al tamaño de la más grande
    pendiente: Optional[float] = None   # inclinación de la patente (dy/dx); None si no se pudo calcular
    proyectadas: List[Proyeccion] = field(default_factory=list)
    estado: str = "insuficiente"        # "completo" | "proyectado" | "insuficiente"
    giro: Optional["Giro"] = None       # datos del giro (None si ROTAR=False o no hay >= 2 caracteres)
    rotado: Optional[np.ndarray] = None            # recorte color girado
    estirado_rotado: Optional[np.ndarray] = None   # recorte girado en grises (contraste estirado)
    binaria_rotada: Optional[np.ndarray] = None    # el girado, binarizado con UMBRAL_BINARIO (solo para mostrar)
    placa: Optional[tuple] = None       # caja negra de la patente (x0, y0, x1, y1), en el recorte girado si hay giro
    punto: Optional[tuple] = None       # (x, y) centro de la patente en la foto original (etapa 1)

    def cajas_finales(self):
        """Todas las cajas de la patente (detectadas + proyectadas) como tuplas
        (x0, y0, x1, y1, tipo), ordenadas de izquierda a derecha."""
        cajas = [(c.x, c.y, c.x + c.w, c.y + c.h, "detectada") for c in self.encontradas]
        cajas += [(p.x0, p.y0, p.x1, p.y1, "proyectada") for p in self.proyectadas]
        return sorted(cajas)


@dataclass
class Giro:
    """Giro que endereza la patente y lo que les pasa a las cajas cuando se gira la imagen."""
    angulo: float                       # grados; > 0 = se gira en sentido antihorario (patente que baja hacia la derecha)
    matriz: np.ndarray                  # matriz afín 2x3 del giro (la misma para la imagen y para las cajas)
    cajas: list                         # cajas ANTES de girar: (x0, y0, x1, y1, tipo)
    poligonos: list                     # esquinas de cada caja DESPUÉS de girar (array 4x2): cuadriláteros inclinados
    centros: np.ndarray                 # (n, 2) centro de cada caja antes de girar
    centros_girados: np.ndarray         # (n, 2) centro de cada caja después de girar
    cajas_giradas: list                 # cajas rectas del mismo tamaño centradas en los centros girados (x0, y0, x1, y1, tipo)


# ============================================================
# ETAPA 1: LOCALIZAR LA PATENTE Y RECORTARLA
# ============================================================

def mapa_transiciones(negro: np.ndarray) -> np.ndarray:
    """Marca con 1 los píxeles donde, yendo de izquierda a derecha, se pasa de negro a claro.

    Es decir: el píxel de la izquierda es negro y el actual no. Cada trazo vertical de una
    letra clara sobre la caja negra deja un 1 en su borde izquierdo. Una zona con muchos
    de estos bordes juntos es muy probablemente una patente.

    negro: matriz de bools (True = píxel negro).
    """
    saltos = np.zeros(negro.shape, np.uint8)
    saltos[:, 1:] = negro[:, :-1] & ~negro[:, 1:]
    return saltos


def detectar_punto(img: np.ndarray, recentrar: bool = True):
    """Devuelve (x, y, W, H): un punto en el centro de la patente y el tamaño de la ventana
    de búsqueda, todo en coordenadas de la foto original.

    Pasos:
      1) Descartar los márgenes izquierdo, derecho y superior (nunca hay patentes).
      2) Marcar los píxeles negros (brillo V < V_NEGRO).
      3) Marcar los saltos negro -> claro (bordes izquierdos de los trazos de las letras).
      4) Limpiar: la erosión vertical elimina saltos aislados (ruido) y deja los que
         forman trazos verticales de al menos ALTO_MIN_TRAZO px; la dilatación los
         vuelve a engrosar.
      5) Promedio móvil: para cada posición, qué fracción de su ventana son saltos.
         La ventana con el mayor valor es la candidata a patente.
      6) Recentrar: promediar la posición de todos los saltos en una zona de 2W x 2H
         alrededor del ganador, para que el punto caiga en medio de los caracteres.
    """
    h_img, w_img = img.shape[:2]

    # 1) zona útil
    x0, x1 = int(MARGEN_IZQ * w_img), int((1 - MARGEN_DER) * w_img)
    y0 = int(MARGEN_SUP * h_img)
    roi = img[y0:, x0:x1]

    # tamaño de la ventana: fracción de la foto ORIGINAL (~ el tamaño de una patente)
    W, H = int(VENTANA_W * w_img), int(VENTANA_H * h_img)

    # 2) píxeles negros: solo importa el brillo (canal V de HSV)
    negro = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2] < V_NEGRO

    # 3) y 4) saltos y limpieza
    saltos = mapa_transiciones(negro)
    saltos = cv2.erode(saltos, np.ones((ALTO_MIN_TRAZO, 1), np.uint8))   # kernel vertical
    saltos = cv2.dilate(saltos, np.ones((3, 1), np.uint8))

    # 5) cantidad de saltos por ventana. BORDER_CONSTANT: fuera de la imagen cuenta como 0.
    cantidad = cv2.blur(saltos.astype(np.float32), (W, H), borderType=cv2.BORDER_CONSTANT)
    cy, cx = np.unravel_index(np.argmax(cantidad), cantidad.shape)   # posición del máximo

    # 6) recentrar
    if recentrar:
        ya, xa = max(0, cy - H), max(0, cx - W)
        ys, xs = np.nonzero(saltos[ya:cy + H, xa:cx + W])
        cy, cx = int(ys.mean()) + ya, int(xs.mean()) + xa

    # volver a coordenadas de la foto original
    return int(cx) + x0, int(cy) + y0, W, H


def zona_recorte(img: np.ndarray, punto, frac_w: float = RECORTE_W, frac_h: float = RECORTE_H):
    """Esquina superior izquierda (x0, y0), ancho y alto del recorte centrado en `punto`.
    Si el punto está cerca del borde de la foto, el recorte se desplaza (no se achica)."""
    h_img, w_img = img.shape[:2]
    cw, ch = int(frac_w * w_img), int(frac_h * h_img)
    x0 = min(max(0, punto[0] - cw // 2), w_img - cw)
    y0 = min(max(0, punto[1] - ch // 2), h_img - ch)
    return x0, y0, cw, ch


def recortar(img: np.ndarray, punto) -> np.ndarray:
    """Recorte rectangular de RECORTE_W x RECORTE_H (fracción de la foto) centrado en `punto`."""
    x0, y0, cw, ch = zona_recorte(img, punto)
    return img[y0:y0 + ch, x0:x0 + cw]


# ============================================================
# ETAPA 2: BINARIZAR Y BUSCAR CANDIDATOS A CARACTERES
# ============================================================

def preparar_binaria(recorte: np.ndarray, umbral: int = UMBRAL_BINARIO):
    """Pasa el recorte a grises, estira el contraste y lo binariza.

    Estirar el contraste: el valor más oscuro pasa a 0 y el más claro a 255 (el resto
    proporcional). Así el umbral fijo funciona aunque la foto sea oscura o clara.
    Devuelve (estirado, binaria); en `binaria`, 255 = claro (> umbral) y 0 = oscuro.
    """
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    estirado = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    binaria = np.where(estirado > umbral, 255, 0).astype(np.uint8)
    return estirado, binaria


def componentes(binaria: np.ndarray) -> List[Caja]:
    """Busca las zonas blancas conectadas ("componentes conexas") que podrían ser un carácter.

    Se descartan las que:
      * tienen menos de AREA_MIN píxeles (ruido),
      * miden menos del 20% o más del 70% del alto del recorte (ALTO_REL),
      * son más anchas que altas (un carácter siempre es más alto que ancho).
    """
    alto_recorte = binaria.shape[0]
    n, _, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=CONECTIVIDAD)
    cajas = []
    for i in range(1, n):                      # la componente 0 es el fondo
        x, y, w, h, area = stats[i]
        if (area >= AREA_MIN
                and ALTO_REL[0] * alto_recorte <= h <= ALTO_REL[1] * alto_recorte
                and w <= h):
            cajas.append(Caja(int(x), int(y), int(w), int(h)))
    return cajas


def preparar_foto(ruta: str):
    """Lee la foto, localiza la patente, la recorta y la binariza.
    Devuelve (punto, recorte, estirado, binaria); `punto` es el centro de la patente en la foto."""
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    cx, cy, _, _ = detectar_punto(img)
    recorte = recortar(img, (cx, cy))
    estirado, binaria = preparar_binaria(recorte)
    return (cx, cy), recorte, estirado, binaria


# ============================================================
# ETAPA 3: ELEGIR LOS CARACTERES, IGUALARLOS Y DEDUCIR LOS QUE FALTAN
# ============================================================

def cajas_de_area_similar(candidatas: List[Caja], tol: float = TOL_AREA, n: int = N_CARACTERES):
    """Elige las cajas que se parecen entre sí en área: son los caracteres de la patente.

    Cada caja "vota" por las demás cuyas áreas difieren menos de `tol` (relativo).
    El área con más votos es la de referencia. Si hay más de `n` parecidas, se dejan
    las `n` más cercanas a la referencia.
    Devuelve las cajas ordenadas de izquierda a derecha.
    """
    if not candidatas:
        return []
    areas = np.array([c.area for c in candidatas], dtype=float)
    votos = [(np.abs(areas - a) <= tol * a).sum() for a in areas]
    ref = areas[int(np.argmax(votos))]
    grupo = [c for c in candidatas if abs(c.area - ref) <= tol * ref]
    grupo = sorted(grupo, key=lambda c: abs(c.area - ref))[:n]
    return sorted(grupo, key=lambda c: c.cx)


def igualar_cajas(cajas: List[Caja]) -> List[Caja]:
    """Fuerza que todas las cajas tengan el tamaño de la más grande (la de mayor área).

    Algunas cajas quedan recortadas porque no abarcan todo el carácter. Cada caja nueva
    se construye con el ancho y alto de la más grande, centrada en el centro de la original.
    """
    if not cajas:
        return []
    mayor = max(cajas, key=lambda c: c.area)
    return [Caja(int(round(c.cx - mayor.w / 2)), int(round(c.cy - mayor.h / 2)), mayor.w, mayor.h)
            for c in cajas]


def pendiente_entre(a: Caja, b: Caja) -> Optional[float]:
    """Inclinación (dy/dx) de la recta que une los centros de dos cajas. None si están en la misma x."""
    if abs(b.cx - a.cx) < 1e-6:
        return None
    return (b.cy - a.cy) / (b.cx - a.cx)


def altura_techo(ref: Caja, pendiente: float, x: float) -> float:
    """Altura (y) en la posición x de la recta con la inclinación de la patente que pasa por
    la esquina SUPERIOR izquierda de la caja `ref`. Es el "techo" de la fila de caracteres."""
    return ref.y + pendiente * (x - ref.x)


def altura_piso(ref: Caja, pendiente: float, x: float) -> float:
    """Igual que altura_techo, pero por la esquina INFERIOR derecha de `ref`: el "piso"."""
    return ref.y + ref.h + pendiente * (x - (ref.x + ref.w))


def proyectar_lado(lado: List[Caja], mitad: float, s: int, margen: float = MARGEN_PROYECCION):
    """Deduce dónde está el 3.er carácter de una mitad de la patente en la que solo hay 2.

    Una patente tiene 3 caracteres a cada lado de su centro (la línea del 50%, `mitad`),
    separados por un paso constante. Con 2 detectados hay 3 casos:

      * Las dos cajas están separadas (el hueco entre ellas es mayor que su ancho):
            falta la que está EN EL MEDIO.
      * Son vecinas y hay lugar entre la más cercana al centro y la línea del 50%
        (su distancia a la línea supera `margen` veces el paso):
            falta la que está HACIA EL CENTRO.
      * Son vecinas y no hay lugar:
            falta la última, EN EL EXTREMO.

    `s` es +1 para la mitad derecha y -1 para la izquierda: se mide la distancia "hacia
    afuera" desde la línea, así ambas mitades se tratan igual.

    Devuelve (cx, ancho, alto, motivo) de la caja faltante, o None si `lado` no tiene 2 cajas.
    """
    if len(lado) != 2:
        return None
    # A = caja más cercana a la línea del 50%; B = la más lejana
    A, B = sorted(lado, key=lambda c: s * (c.cx - mitad))
    uA, uB = s * (A.cx - mitad), s * (B.cx - mitad)   # distancias a la línea, hacia afuera
    paso = uB - uA                                    # distancia entre centros
    hueco = (uB - B.w / 2) - (uA + A.w / 2)           # espacio libre entre los bordes de A y B
    ancho, alto = (A.w + B.w) / 2, (A.h + B.h) / 2

    if hueco >= ancho:
        u_nuevo, motivo = (uA + uB) / 2, "en el medio"
    elif uA > margen * paso:
        u_nuevo, motivo = uA - paso, "hacia el centro"
    else:
        u_nuevo, motivo = uB + paso, "en el extremo"
    return mitad + s * u_nuevo, ancho, alto, motivo


def completar_caracteres(encontradas: List[Caja], ancho_recorte: int):
    """Con los caracteres encontrados, calcula la inclinación de la patente y proyecta
    los que faltan en cada mitad que tenga exactamente 2.

    Devuelve (pendiente, [Proyeccion, ...]). Con menos de 2 cajas no hay inclinación: (None, []).
    """
    if len(encontradas) < 2:
        return None, []
    primera, ultima = encontradas[0], encontradas[-1]
    pend = pendiente_entre(primera, ultima)
    if pend is None:
        return None, []

    mitad = ancho_recorte / 2
    lados = (("izq", -1, [c for c in encontradas if c.cx < mitad]),
             ("der", +1, [c for c in encontradas if c.cx >= mitad]))
    proyectadas = []
    for nombre, s, cajas in lados:
        r = proyectar_lado(cajas, mitad, s)
        if r is None:
            continue
        cx, ancho, _, motivo = r
        xl, xr = cx - ancho / 2, cx + ancho / 2
        # La caja es un rectángulo "recto" (como las detectadas): la esquina superior izquierda
        # se apoya en la recta del techo y la inferior derecha en la del piso.
        proyectadas.append(Proyeccion(nombre, motivo,
                                      xl, altura_techo(primera, pend, xl),
                                      xr, altura_piso(primera, pend, xr)))
    return pend, proyectadas


# ============================================================
# ETAPA 4: ENDEREZAR (GIRAR EL RECORTE Y LAS CAJAS)
# ============================================================

def angulo_por_centros(encontradas: List[Caja]) -> Optional[float]:
    """Ángulo (grados) de la recta que une los centros de la primera y la última caja detectada.
    Positivo = la fila baja hacia la derecha. None si hay menos de 2 cajas o están en la misma x."""
    if len(encontradas) < 2:
        return None
    m = pendiente_entre(encontradas[0], encontradas[-1])
    return None if m is None else float(np.degrees(np.arctan(m)))


def matriz_giro(angulo: float, ancho: int, alto: int) -> np.ndarray:
    """Matriz afín 2x3 de un giro de `angulo` grados alrededor del centro de una imagen ancho x alto.
    Con angulo > 0 el giro es antihorario, que es lo que endereza una fila que baja hacia la derecha."""
    return cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo, 1.0)


def girar(img: np.ndarray, matriz: np.ndarray) -> np.ndarray:
    """Aplica el giro a una imagen. Mantiene el tamaño; los bordes vacíos se rellenan con GIRO_BORDE."""
    alto, ancho = img.shape[:2]
    return cv2.warpAffine(img, matriz, (ancho, alto), flags=GIRO_INTERPOLACION, borderMode=GIRO_BORDE)


def transformar_puntos(matriz: np.ndarray, puntos) -> np.ndarray:
    """Lleva puntos (x, y) de la imagen original a la girada con la MISMA matriz que `girar`.
    Devuelve un array (n, 2)."""
    P = np.asarray(puntos, dtype=np.float64).reshape(-1, 2)
    return P @ matriz[:, :2].T + matriz[:, 2]


def esquinas(x0: float, y0: float, x1: float, y1: float) -> list:
    """Esquinas de un rectángulo en orden: superior izquierda, superior derecha, inferior derecha, inferior izquierda."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def calcular_giro(encontradas: List[Caja], cajas: list, ancho: int, alto: int) -> Optional[Giro]:
    """Calcula el giro y la nueva posición de todas las cajas.

    encontradas: las cajas DETECTADAS (de ellas sale el ángulo).
    cajas: todas las cajas finales (detectadas + proyectadas) como (x0, y0, x1, y1, tipo).
    Devuelve None si no se puede calcular el ángulo (menos de 2 cajas detectadas).
    """
    angulo = angulo_por_centros(encontradas)
    if angulo is None or not cajas:
        return None
    M = matriz_giro(angulo, ancho, alto)
    centros = np.array([((a + c) / 2, (b + d) / 2) for a, b, c, d, _ in cajas])
    centros_g = transformar_puntos(M, centros)
    poligonos = [transformar_puntos(M, esquinas(a, b, c, d)) for a, b, c, d, _ in cajas]
    giradas = [(cx - (c - a) / 2, cy - (d - b) / 2, cx + (c - a) / 2, cy + (d - b) / 2, tipo)
               for (a, b, c, d, tipo), (cx, cy) in zip(cajas, centros_g)]
    return Giro(angulo, M, cajas, poligonos, centros, centros_g, giradas)


# ============================================================
# ETAPA 5: SEGMENTAR LA PLACA (LA CAJA NEGRA DE LA PATENTE)
# ============================================================

def segmentar_placa(oscuro: np.ndarray, cajas: list):
    """Delimita la caja negra de la patente a partir de la fila de caracteres.

    oscuro: matriz de bools (True = píxel oscuro) del recorte, idealmente ya enderezado.
    cajas: cajas de los caracteres como (x0, y0, x1, y1, tipo).

    Se parte del rectángulo que encierra a los 6 caracteres y se lo agranda de a un píxel
    hacia cada lado mientras la columna (o fila) nueva sea mayormente oscura (más de
    FRAC_OSCURO): se sigue dentro de la caja negra. Al llegar al marco claro de la patente,
    se frena. Por las dudas (fondo oscuro alrededor de la patente) nunca se agranda más de
    un ancho de carácter hacia los costados ni más de un cuarto de alto hacia arriba/abajo.
    Devuelve (x0, y0, x1, y1).
    """
    alto, ancho = oscuro.shape
    x0 = max(0, int(min(c[0] for c in cajas)))           # sin salirse del recorte
    y0 = max(0, int(min(c[1] for c in cajas)))
    x1 = min(ancho - 1, int(max(c[2] for c in cajas)))
    y1 = min(alto - 1, int(max(c[3] for c in cajas)))
    ancho_car = cajas[0][2] - cajas[0][0]     # todas las cajas miden lo mismo (igualar_cajas)
    alto_car = cajas[0][3] - cajas[0][1]

    # límites: hasta dónde se puede agrandar como máximo
    x_min, x_max = max(0, int(x0 - ancho_car)), min(ancho - 1, int(x1 + ancho_car))
    y_min, y_max = max(0, int(y0 - alto_car / 4)), min(alto - 1, int(y1 + alto_car / 4))

    while x0 > x_min and oscuro[y0:y1, x0 - 1].mean() > FRAC_OSCURO:   # hacia la izquierda
        x0 -= 1
    while x1 < x_max and oscuro[y0:y1, x1].mean() > FRAC_OSCURO:       # hacia la derecha
        x1 += 1
    while y0 > y_min and oscuro[y0 - 1, x0:x1].mean() > FRAC_OSCURO:   # hacia arriba
        y0 -= 1
    while y1 < y_max and oscuro[y1, x0:x1].mean() > FRAC_OSCURO:       # hacia abajo
        y1 += 1
    return x0, y0, x1, y1


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def analizar_recorte(recorte: np.ndarray) -> Resultado:
    """Etapas 2, 3, 4 y 5 sobre un recorte ya extraído (la detección se hace SIN rotar)."""
    estirado, binaria = preparar_binaria(recorte)
    candidatas = componentes(binaria)
    originales = cajas_de_area_similar(candidatas)
    encontradas = igualar_cajas(originales)
    res = Resultado(recorte, estirado, binaria, candidatas, originales, encontradas)

    if len(encontradas) == N_CARACTERES:
        res.estado = "completo"
    elif len(encontradas) >= 2:
        res.pendiente, res.proyectadas = completar_caracteres(encontradas, binaria.shape[1])
        res.estado = "proyectado" if res.proyectadas else "insuficiente"

    # Etapa 4: con las cajas ya definidas, girar el recorte y las cajas con la misma matriz
    if ROTAR:
        alto, ancho = binaria.shape
        res.giro = calcular_giro(encontradas, res.cajas_finales(), ancho, alto)
        if res.giro is not None:
            res.rotado = girar(recorte, res.giro.matriz)
            res.estirado_rotado = girar(estirado, res.giro.matriz)
            res.binaria_rotada = np.where(res.estirado_rotado > UMBRAL_BINARIO, 255, 0).astype(np.uint8)

    # Etapa 5: la caja negra de la patente (sobre el recorte girado, si se pudo girar)
    if res.giro is not None:
        res.placa = segmentar_placa(res.estirado_rotado <= UMBRAL_BINARIO, res.giro.cajas_giradas)
    elif res.cajas_finales():
        res.placa = segmentar_placa(estirado <= UMBRAL_BINARIO, res.cajas_finales())
    return res


def analizar_foto(ruta: str) -> Resultado:
    """Corre todo el flujo sobre una foto y devuelve un Resultado.

    estado:
      "completo"     -> se encontraron los 6 caracteres directamente.
      "proyectado"   -> faltaban; se dedujo la posición de los que faltan (ver `proyectadas`).
      "insuficiente" -> muy pocos caracteres para deducir algo.
    Si ROTAR=True y hay >= 2 caracteres detectados, `giro`, `rotado`, `estirado_rotado` y `binaria_rotada`
    traen el recorte enderezado y las cajas giradas.
    """
    punto, recorte, _, _ = preparar_foto(ruta)
    res = analizar_recorte(recorte)
    res.punto = punto
    return res


def informar(ruta: str, r: Resultado):
    """Imprime por consola el resultado de cada etapa para una foto."""
    print(f"\n=== {ruta} ===")
    print(f"  1) Patente localizada: centro en (x={r.punto[0]}, y={r.punto[1]}) de la foto; "
          f"recorte de {r.recorte.shape[1]}x{r.recorte.shape[0]} px")
    print(f"  2) Componentes candidatas a carácter: {len(r.candidatas)}")
    print(f"  3) Caracteres elegidos por área parecida: {len(r.encontradas)}  ->  estado: {r.estado}")
    for q in r.proyectadas:
        print(f"       deducido en la mitad {q.lado}: falta {q.motivo}")
    if r.giro is not None:
        print(f"  4) Inclinación corregida: {r.giro.angulo:+.1f}°")
    else:
        print("  4) Sin giro (menos de 2 caracteres detectados)")
    if r.placa is not None:
        x0, y0, x1, y1 = r.placa
        print(f"  5) Placa segmentada: x={x0}..{x1}, y={y0}..{y1} ({x1 - x0}x{y1 - y0} px)")
    print(f"  Caracteres segmentados: {len(r.cajas_finales())} de {N_CARACTERES}")


# ============================================================
# VISUALIZACIÓN: UNA FIGURA (PESTAÑA) POR FOTO
# ============================================================

COLOR_CANDIDATA = (255, 200, 0)    # celeste: componentes candidatas
COLOR_ORIGINAL = (0, 255, 255)     # amarillo: cajas elegidas antes de igualarlas
COLOR_ENCONTRADA = (0, 255, 0)     # verde: caracteres elegidos (ya igualados)
COLOR_PROYECTADA = (0, 165, 255)   # naranja: caracteres deducidos
COLOR_PLACA = (0, 0, 255)          # rojo: placa segmentada
COLOR_CENTROS, COLOR_TECHO, COLOR_PISO = (0, 0, 255), (255, 255, 0), (255, 0, 255)  # rectas (BGR)


def _ampliar(img: np.ndarray, factor: int) -> np.ndarray:
    """Agranda una imagen (sin suavizar) y la devuelve en BGR para poder dibujar en color."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def _dibujar_caja(vis, caja: Caja, color, factor):
    cv2.rectangle(vis, (caja.x * factor, caja.y * factor),
                  ((caja.x + caja.w) * factor, (caja.y + caja.h) * factor), color, 1)


def _trazar_recta(vis, punto, pendiente, ancho, factor, color):
    """Dibuja de borde a borde la recta con la pendiente dada que pasa por `punto`."""
    x0, y0 = punto
    y_izq = y0 + pendiente * (0 - x0)
    y_der = y0 + pendiente * ((ancho - 1) - x0)
    cv2.line(vis, (0, int(y_izq * factor)), ((ancho - 1) * factor, int(y_der * factor)), color, 1)


def _rgb(vis):
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _pt(x, y, factor):
    """Coordenadas del recorte -> píxeles de la imagen ampliada."""
    return int(round(x * factor)), int(round(y * factor))


def _poligono(vis, esq, color, factor, grosor=1):
    cv2.polylines(vis, [np.array([_pt(x, y, factor) for x, y in esq], np.int32)], True, color, grosor)


def ver_foto(ruta: str, factor: int = 6, guardar: Optional[str] = None):
    """Una figura (una pestaña/ventana) para UNA foto.

    Fila de arriba (4 paneles):
      0) la foto original con la zona recortada (rojo) y el centro de la patente encontrado,
      1) el recorte de la patente con las cajas DETECTADAS (verde),
      2) el mismo recorte con las rectas (50% blanca, centros roja, techo celeste, piso magenta)
         y las cajas PROYECTADAS (naranja); si ya hay 6 detectadas no hay nada que proyectar,
      3) el recorte GIRADO con las cajas finales ya giradas (verde = detectada, naranja = proyectada)
         y la PLACA segmentada (rojo).
    Filas de abajo: cada carácter recortado de la imagen girada, de izquierda a derecha
    (en grises y en binario), con su tipo (detectada/proyectada) y sus valores (x, y, ancho x alto).

    Si se indica `guardar` (ruta de archivo), se guarda la figura en vez de dejarla abierta.
    """
    r = analizar_foto(ruta)
    informar(ruta, r)
    alto, ancho = r.binaria.shape
    g = r.giro

    # --- panel 0: foto original con la zona recortada (etapa 1)
    vis0 = cv2.imread(ruta, cv2.IMREAD_COLOR)
    x0, y0, cw, ch = zona_recorte(vis0, r.punto)
    cv2.rectangle(vis0, (x0, y0), (x0 + cw, y0 + ch), COLOR_PLACA, 2)
    cv2.circle(vis0, r.punto, 3, COLOR_PLACA, -1)

    # --- panel 1: recorte con las cajas detectadas
    vis1 = _ampliar(r.recorte, factor)
    for c in r.encontradas:
        _dibujar_caja(vis1, c, COLOR_ENCONTRADA, factor)

    # --- panel 2: rectas y cajas proyectadas
    vis2 = _ampliar(r.recorte, factor)
    for c in r.encontradas:
        _dibujar_caja(vis2, c, COLOR_ENCONTRADA, factor)
    if r.estado == "completo":
        resumen = f"{N_CARACTERES} detectadas: no hace falta proyectar"
    elif r.estado == "insuficiente":
        resumen = f"{len(r.encontradas)} detectadas: no alcanza para deducir"
    else:
        mitad = ancho / 2
        cv2.line(vis2, (int(mitad * factor), 0), (int(mitad * factor), alto * factor), (255, 255, 255), 1)
        ref = r.encontradas[0]
        _trazar_recta(vis2, (ref.cx, ref.cy), r.pendiente, ancho, factor, COLOR_CENTROS)
        _trazar_recta(vis2, (ref.x, ref.y), r.pendiente, ancho, factor, COLOR_TECHO)
        _trazar_recta(vis2, (ref.x + ref.w, ref.y + ref.h), r.pendiente, ancho, factor, COLOR_PISO)
        for q in r.proyectadas:
            cv2.rectangle(vis2, (int(q.x0 * factor), int(q.y0 * factor)),
                          (int(q.x1 * factor), int(q.y1 * factor)), COLOR_PROYECTADA, 1)
        resumen = " | ".join(f"{q.lado}: falta {q.motivo}" for q in r.proyectadas) or "ninguna mitad con 2 caracteres"

    # --- panel 3: recorte girado con las cajas finales giradas
    vis3 = None
    if g is not None:
        vis3 = _ampliar(r.rotado, factor)
        for x0, y0, x1, y1, tipo in g.cajas_giradas:
            color = COLOR_PROYECTADA if tipo == "proyectada" else COLOR_ENCONTRADA
            _poligono(vis3, esquinas(x0, y0, x1, y1), color, factor)
        if r.placa is not None:
            _poligono(vis3, esquinas(*r.placa), COLOR_PLACA, factor, grosor=3)

    # --- cajas con las que se recorta cada carácter: de la imagen girada si hay giro
    if g is not None:
        cajas, gris_ch, bin_ch = g.cajas_giradas, r.estirado_rotado, r.binaria_rotada
        origen = "imagen girada"
    else:
        cajas, gris_ch, bin_ch = r.cajas_finales(), r.estirado, r.binaria
        origen = "recorte sin girar"

    # --- figura
    fig = plt.figure(figsize=(16, 7.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(ruta)                       # título de la pestaña/ventana
    # 12 columnas: arriba 4 paneles de 3 columnas; abajo 6 caracteres de 2 columnas
    gs = fig.add_gridspec(3, 12, height_ratios=[1.15, 1, 1])

    def panel(spec, img, titulo, fs=10):
        ax = fig.add_subplot(spec)
        if img is None:
            ax.text(0.5, 0.5, "sin giro", ha="center", va="center")
        else:
            ax.imshow(_rgb(img))
        ax.set_title(titulo, fontsize=fs); ax.axis("off")

    ang = f"{g.angulo:+.1f}°" if g is not None else "-"
    panel(gs[0, 0:3], vis0, "foto: zona recortada (rojo)")
    panel(gs[0, 3:6], vis1, f"{len(r.encontradas)} cajas detectadas")
    panel(gs[0, 6:9], vis2, resumen, 9)
    panel(gs[0, 9:12], vis3, f"girado {ang}: cajas finales y placa (rojo)")

    for j in range(N_CARACTERES):
        ax_g, ax_b = fig.add_subplot(gs[1, 2 * j:2 * j + 2]), fig.add_subplot(gs[2, 2 * j:2 * j + 2])
        ax_g.axis("off"); ax_b.axis("off")
        if j >= len(cajas):
            continue                                                    # menos de 6 cajas: dejar vacío
        x0, y0, x1, y1, tipo = cajas[j]
        a, b = max(0, int(round(x0))), max(0, int(round(y0)))           # recortar sin salirse del recorte
        c, d = min(ancho, int(round(x1))), min(alto, int(round(y1)))
        ax_g.imshow(gris_ch[b:d, a:c], cmap="gray", vmin=0, vmax=255)
        ax_b.imshow(bin_ch[b:d, a:c], cmap="gray", vmin=0, vmax=255)
        ax_g.set_title(f"{j + 1}: {tipo}\nx={a} y={b}  {c - a}x{d - b}", fontsize=9,
                       color="darkorange" if tipo == "proyectada" else "black")

    fig.suptitle(f"{ruta[-9:]}  -  estado: {r.estado}  -  caracteres de la {origen}", fontsize=13)
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=100)
        plt.close(fig)


def ver_resultados(rutas: List[str], factor: int = 6, guardar_prefijo: Optional[str] = None):
    """Una figura por foto (en un script, una ventana/pestaña por foto, todas abiertas a la vez;
    en un notebook, una debajo de otra).
    Con `guardar_prefijo="salida/patente"` guarda salida/patente_01.png, _02.png, ... en vez de mostrar."""
    for i, ruta in enumerate(rutas, start=1):
        guardar = f"{guardar_prefijo}_{i:02d}.png" if guardar_prefijo else None
        ver_foto(ruta, factor, guardar)
    if not guardar_prefijo:
        plt.show()


# ============================================================
# EJECUCIÓN DIRECTA:  python patentes.py [foto1.png foto2.png ...]
# ============================================================

if __name__ == "__main__":
    # rutas_fotos = sys.argv[1:] or [f"images/Patentes/img{i:02d}.png" for i in range(1, 13)]
    rutas_fotos = sys.argv[1:] or [f"images/Patentes/img{i:02d}.png" for i in range(1, 13)]
    ver_resultados(rutas_fotos)