"""
patentes.py
===========
Localización y segmentación de los 6 caracteres de patentes argentinas (formato
viejo: 3 letras + 3 números, caja negra con caracteres claros) en fotos de 640x360.

El flujo completo tiene 3 etapas (cada una es una sección de este archivo):

  1) LOCALIZAR la patente en la foto y recortarla.
        detectar_punto()  ->  recortar()
  2) BINARIZAR el recorte (blanco/negro) y buscar CANDIDATOS a caracteres
     con componentes conexas.
        preparar_foto()   ->  componentes()
  3) ELEGIR los caracteres, igualar sus cajas y, si faltan, DEDUCIR dónde están
     usando la geometría de la patente (alineación, separación constante,
     mitad izquierda/derecha).
        cajas_de_area_similar()  ->  igualar_cajas()  ->  proyectar_lado()

Punto de entrada:   analizar_foto(ruta)   -> Resultado
Para verlo:         python patentes.py     (abre una ventana/pestaña por foto)
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

    def cajas_finales(self):
        """Todas las cajas de la patente (detectadas + proyectadas) como tuplas
        (x0, y0, x1, y1, tipo), ordenadas de izquierda a derecha."""
        cajas = [(c.x, c.y, c.x + c.w, c.y + c.h, "detectada") for c in self.encontradas]
        cajas += [(p.x0, p.y0, p.x1, p.y1, "proyectada") for p in self.proyectadas]
        return sorted(cajas)


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


def recortar(img: np.ndarray, punto, frac_w: float = RECORTE_W, frac_h: float = RECORTE_H) -> np.ndarray:
    """Recorte rectangular de (frac_w * ancho) x (frac_h * alto) centrado en `punto`.
    Si el punto está cerca del borde de la foto, el recorte se desplaza (no se achica)."""
    h_img, w_img = img.shape[:2]
    cw, ch = int(frac_w * w_img), int(frac_h * h_img)
    x0 = min(max(0, punto[0] - cw // 2), w_img - cw)
    y0 = min(max(0, punto[1] - ch // 2), h_img - ch)
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
    Devuelve (recorte, estirado, binaria)."""
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    cx, cy, _, _ = detectar_punto(img)
    recorte = recortar(img, (cx, cy))
    estirado, binaria = preparar_binaria(recorte)
    return recorte, estirado, binaria


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
# FUNCIÓN PRINCIPAL
# ============================================================

def analizar_foto(ruta: str) -> Resultado:
    """Corre todo el flujo sobre una foto y devuelve un Resultado.

    estado:
      "completo"     -> se encontraron los 6 caracteres directamente.
      "proyectado"   -> faltaban; se dedujo la posición de los que faltan (ver `proyectadas`).
      "insuficiente" -> muy pocos caracteres para deducir algo.
    """
    recorte, estirado, binaria = preparar_foto(ruta)
    candidatas = componentes(binaria)
    originales = cajas_de_area_similar(candidatas)
    encontradas = igualar_cajas(originales)
    res = Resultado(recorte, estirado, binaria, candidatas, originales, encontradas)

    if len(encontradas) == N_CARACTERES:
        res.estado = "completo"
    elif len(encontradas) >= 2:
        res.pendiente, res.proyectadas = completar_caracteres(encontradas, binaria.shape[1])
        res.estado = "proyectado" if res.proyectadas else "insuficiente"
    return res


# ============================================================
# VISUALIZACIÓN: UNA FIGURA (PESTAÑA) POR FOTO
# ============================================================

COLOR_CANDIDATA = (255, 200, 0)    # celeste: componentes candidatas
COLOR_ORIGINAL = (0, 255, 255)     # amarillo: cajas elegidas antes de igualarlas
COLOR_ENCONTRADA = (0, 255, 0)     # verde: caracteres elegidos (ya igualados)
COLOR_PROYECTADA = (0, 165, 255)   # naranja: caracteres deducidos
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


def ver_foto(ruta: str, factor: int = 6, guardar: Optional[str] = None):
    """Una figura (una pestaña/ventana) para UNA foto.

    Fila de arriba (3 paneles):
      1) el recorte de la patente,
      2) la binaria con todas las candidatas (celeste),
      3) el resultado: cajas originales (amarillo), igualadas (verde), línea del 50% (blanca),
         rectas de centros (roja), techo (celeste) y piso (magenta), cajas deducidas (naranja).
    Filas de abajo: cada carácter por separado, de izquierda a derecha (en grises y en binario),
    con su tipo (detectada/proyectada) y sus valores (x, y, ancho x alto) dentro del recorte.

    Si se indica `guardar` (ruta de archivo), se guarda la figura en vez de dejarla abierta.
    """
    r = analizar_foto(ruta)
    alto, ancho = r.binaria.shape

    # --- panel 2: binaria con candidatas
    vis_bin = _ampliar(r.binaria, factor)
    for c in r.candidatas:
        _dibujar_caja(vis_bin, c, COLOR_CANDIDATA, factor)

    # --- panel 3: resultado
    vis = _ampliar(r.estirado, factor)
    for c in r.originales:
        _dibujar_caja(vis, c, COLOR_ORIGINAL, factor)
    for c in r.encontradas:
        _dibujar_caja(vis, c, COLOR_ENCONTRADA, factor)

    if r.estado == "completo":
        resumen = f"{N_CARACTERES} detectadas -> listo"
    elif r.estado == "insuficiente":
        resumen = f"{len(r.encontradas)} detectadas: no alcanza para deducir"
    else:
        mitad = ancho / 2
        cv2.line(vis, (int(mitad * factor), 0), (int(mitad * factor), alto * factor), (255, 255, 255), 1)
        ref = r.encontradas[0]
        _trazar_recta(vis, (ref.cx, ref.cy), r.pendiente, ancho, factor, COLOR_CENTROS)
        _trazar_recta(vis, (ref.x, ref.y), r.pendiente, ancho, factor, COLOR_TECHO)
        _trazar_recta(vis, (ref.x + ref.w, ref.y + ref.h), r.pendiente, ancho, factor, COLOR_PISO)
        for p in r.proyectadas:
            cv2.rectangle(vis, (int(p.x0 * factor), int(p.y0 * factor)),
                          (int(p.x1 * factor), int(p.y1 * factor)), COLOR_PROYECTADA, 1)
        resumen = f"{len(r.encontradas)} detectadas | " + " | ".join(
            f"{p.lado}: falta {p.motivo}" for p in r.proyectadas)

    # --- figura
    cajas = r.cajas_finales()
    fig = plt.figure(figsize=(14, 7.5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(ruta)                       # título de la pestaña/ventana
    gs = fig.add_gridspec(3, N_CARACTERES, height_ratios=[1.15, 1, 1])

    ax = fig.add_subplot(gs[0, 0:2]); ax.imshow(_rgb(r.recorte)); ax.set_title("recorte de la patente", fontsize=11); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2:4]); ax.imshow(_rgb(vis_bin)); ax.set_title(f"{len(r.candidatas)} candidatas", fontsize=11); ax.axis("off")
    ax = fig.add_subplot(gs[0, 4:6]); ax.imshow(_rgb(vis)); ax.set_title(resumen, fontsize=9); ax.axis("off")

    for j in range(N_CARACTERES):
        ax_g, ax_b = fig.add_subplot(gs[1, j]), fig.add_subplot(gs[2, j])
        ax_g.axis("off"); ax_b.axis("off")
        if j >= len(cajas):
            continue                                                    # menos de 6 cajas: dejar vacío
        x0, y0, x1, y1, tipo = cajas[j]
        a, b = max(0, int(round(x0))), max(0, int(round(y0)))           # recortar sin salirse del recorte
        c, d = min(ancho, int(round(x1))), min(alto, int(round(y1)))
        ax_g.imshow(r.estirado[b:d, a:c], cmap="gray", vmin=0, vmax=255)
        ax_b.imshow(r.binaria[b:d, a:c], cmap="gray", vmin=0, vmax=255)
        ax_g.set_title(f"{j + 1}: {tipo}\nx={a} y={b}  {c - a}x{d - b}", fontsize=9,
                       color="darkorange" if tipo == "proyectada" else "black")

    fig.suptitle(f"{ruta[-9:]}  -  estado: {r.estado}", fontsize=13)
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
    rutas_fotos = sys.argv[1:] or [f"tp2_procesamiento_de_imagenes/images/Patentes/img{i:02d}.png" for i in range(1, 13)]
    ver_resultados(rutas_fotos)