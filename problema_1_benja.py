"""
TP2 - Problema 1: Detección y clasificación de monedas y dados.

Pipeline (cada etapa informa por consola y muestra una figura):

    A. Segmentación
        1. Carga y suavizado (filtro gaussiano).
        2. Pasaje a CIELAB: canal L (brillo) y croma (saturación del color).
        3. Binarización: croma alto (monedas doradas/bimetálicas)
                         o brillo alto (monedas plateadas y dados).
                         El umbral de brillo lo elige Otsu (U3).
        4. Morfología: clausura, relleno de huecos (floodFill) y apertura.
        5. Componentes conectados: un objeto por moneda o dado.

    B. Clasificación de monedas
        6. Contorno, área y perímetro de cada objeto -> factor de forma rho.
           rho alto = círculo = moneda;  rho bajo = dado.
        7. Las monedas se clasifican por ÁREA con k-means (U5): se agrupan las
           áreas en 3 grupos (10 centavos / 1 peso / 50 centavos) sin umbrales fijos.

    C. Dados
        8. Dentro de cada dado se buscan los puntos negros (pips) y se tapan los
           agujeritos de los reflejos con una clausura (U6).
        9. Los pips de la cara superior son círculos grandes (área alta y
           rho alto); los de las caras laterales están estirados y son chicos.
           "Grande" se mide contra el pip más grande de ese mismo dado.
           El valor del dado es la cantidad de pips de la cara superior.

Uso:  python problema_1_benja.py   (desde la carpeta del proyecto)
Las figuras se guardan en images/output_benja/
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Parámetros (ajustados sobre images/monedas.jpg)
# =============================================================================

# RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_IMAGEN = 'images/monedas.jpg'
CARPETA_SALIDA = 'images/output_benja/'

# --- Segmentación ------------------------------------------------------------
SIGMA_SUAVIZADO = 2        # Desvío del filtro gaussiano
UMBRAL_CROMA = 4           # Croma > umbral  -> objeto con color (el fondo es gris)
LIMITE_CLARO = 150         # Otsu se calcula solo con los píxeles L > límite (el fondo oscuro no cuenta)
K_CLAUSURA = 31            # Elemento estructurante de la clausura (tapa grietas)
K_APERTURA = 61            # Elemento estructurante de la apertura (saca ruido)

# --- Clasificación de monedas ------------------------------------------------
RHO_MIN_MONEDA = 0.8       # rho > 0.8 -> moneda (círculo);  rho <= 0.8 -> dado
# Áreas medidas (px): 10c ~ 55k-63k | 1 peso ~ 84k-87k | 50c ~ 105k
CLAVES_MONEDAS = ['10c', '1p', '50c']   # De menor a mayor área (k-means usa 3 grupos)

# --- Dados -------------------------------------------------------------------
K_APERTURA_PIPS = 5        # Apertura para limpiar los puntos negros
K_CLAUSURA_PIPS = 15       # Clausura para tapar los reflejos y mordidas dentro de un pip
FRAC_AREA_PIP = 0.5        # Un pip de la cara superior mide al menos esta fracción del pip más grande
RHO_MIN_PIP = 0.65         # ... y circulares (los de las caras laterales estan estirados)

MONEDAS = {
    '10c':  {'nombre': '10 centavos', 'valor': 0.10, 'color': (255, 0, 0)},
    '1p':   {'nombre': '1 peso',      'valor': 1.00, 'color': (0, 160, 0)},
    '50c':  {'nombre': '50 centavos', 'valor': 0.50, 'color': (0, 0, 255)},
}


# =============================================================================
# Funciones auxiliares
# =============================================================================

def elemento(k):
    """Elemento estructurante elíptico de k x k."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def rellenar_huecos(binaria):
    """
    Rellena los huecos internos de una imagen binaria (0/255) con floodFill.
    Se inunda el fondo desde la esquina (0, 0); lo que no se inunda es objeto.
    """
    inundada = binaria.copy()
    mascara = np.zeros((binaria.shape[0] + 2, binaria.shape[1] + 2), np.uint8)
    cv2.floodFill(inundada, mascara, (0, 0), 255)
    return binaria | cv2.bitwise_not(inundada)


def medidas_contorno(obj):
    """
    Devuelve (contorno, area, perimetro, rho) de un objeto binario (0/1).
    rho = 4*pi*A / P^2  (= 1 para un círculo perfecto).
    """
    contornos, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    rho = 4 * np.pi * area / perimetro ** 2
    return cnt, area, perimetro, rho


def mostrar(imagenes, titulos, nombre_archivo, titulo_general=None, cmap='gray'):
    """Muestra varias imágenes en una fila y guarda la figura."""
    fig, axs = plt.subplots(1, len(imagenes), figsize=(6 * len(imagenes), 5.5))
    axs = np.atleast_1d(axs)
    for ax, im, t in zip(axs, imagenes, titulos):
        ax.imshow(im, cmap=cmap if im.ndim == 2 else None)
        ax.set_title(t)
        ax.axis('off')
    if titulo_general:
        fig.suptitle(titulo_general, fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(CARPETA_SALIDA, nombre_archivo), dpi=150, bbox_inches='tight')
    plt.show(block=False)


def etiquetas_a_color(labels, num_labels):
    """Pinta cada componente conectado con un color distinto."""
    labels_color = np.uint8(255 / max(num_labels - 1, 1) * labels)
    im_color = cv2.applyColorMap(labels_color, cv2.COLORMAP_JET)
    im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
    im_color[labels == 0] = 0
    return im_color


# =============================================================================
# A. Segmentación
# =============================================================================

def segmentar(img_rgb):
    """
    Segmenta monedas y dados. Devuelve la máscara binaria final (0/255) y
    las imágenes intermedias para mostrarlas.
    """
    # --- 1. Suavizado ---------------------------------------------------------
    img_suave = cv2.GaussianBlur(img_rgb, (0, 0), SIGMA_SUAVIZADO)

    # --- 2. CIELAB: L (brillo) y croma ----------------------------------------
    # El fondo es gris (croma ~ 0) aunque su brillo cambie de izquierda a
    # derecha: el croma no depende del degradé de iluminación.
    lab = cv2.cvtColor(img_suave, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    croma = np.hypot(lab[..., 1].astype(np.float32) - 128,
                     lab[..., 2].astype(np.float32) - 128)

    # --- 3. Binarización ------------------------------------------------------
    # Monedas doradas y bimetálicas: croma alto.
    # Monedas plateadas y dados: casi sin color, pero muy claros.
    # El umbral de brillo lo calcula Otsu, pero solo con los píxeles claros: si
    # se usara toda la imagen, el fondo (que se aclara hacia la derecha) quedaría
    # mezclado con los objetos.
    umbral_brillo, _ = cv2.threshold(L[L > LIMITE_CLARO].reshape(-1, 1), 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_croma = (croma > UMBRAL_CROMA).astype(np.uint8) * 255
    bin_brillo = (L > umbral_brillo).astype(np.uint8) * 255
    binaria = cv2.bitwise_or(bin_croma, bin_brillo)

    # --- 4. Morfología --------------------------------------------------------
    clausura = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, elemento(K_CLAUSURA))
    rellena = rellenar_huecos(clausura)
    mascara = cv2.morphologyEx(rellena, cv2.MORPH_OPEN, elemento(K_APERTURA))

    etapas = {
        'L': L, 'croma': croma, 'umbral_brillo': umbral_brillo,
        'bin_croma': bin_croma, 'bin_brillo': bin_brillo, 'binaria': binaria,
        'clausura': clausura, 'rellena': rellena, 'mascara': mascara,
    }
    return mascara, etapas


# =============================================================================
# B. Medición y clasificación de monedas
# =============================================================================

def medir_objetos(labels, num_labels):
    """Calcula contorno, área, perímetro y rho de cada componente."""
    objetos = []
    for i in range(1, num_labels):
        obj = (labels == i).astype(np.uint8)
        cnt, area, per, rho = medidas_contorno(obj)
        M = cv2.moments(cnt)
        centro = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        objetos.append({'id': i, 'obj': obj, 'cnt': cnt, 'area': area,
                        'perimetro': per, 'rho': rho, 'centro': centro})
    return objetos


def clasificar_monedas(monedas):
    """
    Clasifica las monedas por su área con k-means (3 grupos). Escribe o['tipo'] en
    cada moneda y devuelve los centros de los grupos y los umbrales de corte
    (punto medio entre grupos vecinos), solo para mostrarlos en el gráfico.
    """
    areas = np.float32([o['area'] for o in monedas]).reshape(-1, 1)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    cv2.setRNGSeed(0)   # misma corrida, mismo resultado
    _, etiquetas, centros = cv2.kmeans(areas, len(CLAVES_MONEDAS), None, criterio, 10,
                                       cv2.KMEANS_PP_CENTERS)
    centros = centros.flatten()
    orden = np.argsort(centros)             # grupo de menor a mayor área
    posicion = np.empty_like(orden)
    posicion[orden] = np.arange(len(orden))  # número de grupo -> posición (0=10c, 1=1p, 2=50c)
    for o, e in zip(monedas, etiquetas.flatten()):
        o['tipo'] = CLAVES_MONEDAS[posicion[e]]
    centros = np.sort(centros)
    return centros, (centros[:-1] + centros[1:]) / 2


def dibujar_texto(img, texto, centro, color, escala=2.6, grosor=7):
    """Texto centrado con contorno blanco para que se lea sobre cualquier fondo."""
    (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    org = (centro[0] - w // 2, centro[1] + h // 2)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), grosor + 6)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor)


# =============================================================================
# C. Dados
# =============================================================================

def contar_pips(gris, dado):
    """
    Cuenta los pips de la cara superior de un dado.

    1. Se rellena la silueta del dado con su casco convexo (la máscara de la
       segmentación puede tener muescas donde hay un pip oscuro en el borde).
    2. Dentro del dado se binarizan los píxeles oscuros con Otsu.
    3. Se obtienen los componentes conectados (candidatos a pip).
    4. Se conservan los que son grandes y redondos (rho alto): son los de la
       cara superior. Los de las caras laterales se ven chicos y estirados.
    """
    # 1. Silueta completa del dado
    hull = cv2.convexHull(dado['cnt'])
    mascara = np.zeros_like(gris)
    cv2.fillPoly(mascara, [hull], 255)

    # 2. Píxeles oscuros dentro del dado (Otsu calculado solo con los píxeles del dado)
    valores = gris[mascara > 0].reshape(-1, 1)
    umbral, _ = cv2.threshold(valores, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    oscuros = ((gris < umbral) & (mascara > 0)).astype(np.uint8) * 255
    oscuros = cv2.morphologyEx(oscuros, cv2.MORPH_OPEN, elemento(K_APERTURA_PIPS))
    # Clausura: tapa los reflejos blancos y las mordidas dentro de cada pip
    oscuros = cv2.morphologyEx(oscuros, cv2.MORPH_CLOSE, elemento(K_CLAUSURA_PIPS))

    # 3 y 4. Componentes conectados y filtro por área y factor de forma.
    # El área mínima es relativa: una fracción del pip redondo más grande del dado.
    n, labels, _, _ = cv2.connectedComponentsWithStats(oscuros)
    candidatos = medir_objetos(labels, n)
    redondos = [c['area'] for c in candidatos if c['rho'] > RHO_MIN_PIP]
    area_min = FRAC_AREA_PIP * max(redondos, default=0)
    pips_top, descartados = [], []
    for c in candidatos:
        if c['area'] >= area_min and c['rho'] > RHO_MIN_PIP:
            pips_top.append(c)
        else:
            descartados.append(c)
    return pips_top, descartados, oscuros, umbral


# =============================================================================
# Programa principal
# =============================================================================

def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # -------------------------------------------------------------------------
    # Carga
    # -------------------------------------------------------------------------
    img_bgr = cv2.imread(RUTA_IMAGEN, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f'No se pudo cargar la imagen: {RUTA_IMAGEN}')
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gris = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
    print(f'Imagen cargada: {img.shape[1]} x {img.shape[0]} px')

    # =========================================================================
    # A. SEGMENTACIÓN
    # =========================================================================
    print('\n=== A. Segmentación de monedas y dados ===')
    mascara, e = segmentar(img)

    mostrar([img, e['L'], e['croma']],
            ['Imagen original', 'Canal L (brillo)', 'Croma (color)'],
            'A1_canales.png', 'A1. Pasaje a CIELAB: el croma ignora el degradé del fondo')
    print(f"Binarización: croma > {UMBRAL_CROMA} o L > {e['umbral_brillo']:.0f} (umbral de brillo por Otsu)")

    mostrar([e['bin_croma'], e['bin_brillo'], e['binaria']],
            [f'Croma > {UMBRAL_CROMA}', f"L > {e['umbral_brillo']:.0f} (Otsu)", 'Unión (OR)'],
            'A2_binarizacion.png', 'A2. Binarización')

    mostrar([e['clausura'], e['rellena'], mascara],
            [f'Clausura {K_CLAUSURA}x{K_CLAUSURA}', 'Relleno de huecos (floodFill)',
             f'Apertura {K_APERTURA}x{K_APERTURA} (máscara final)'],
            'A3_morfologia.png', 'A3. Operaciones morfológicas')

    num_labels, labels, stats, centroides = cv2.connectedComponentsWithStats(mascara)
    print(f'Objetos detectados (monedas + dados): {num_labels - 1}')

    segmentada = img.copy()
    segmentada[mascara == 0] = 0
    mostrar([etiquetas_a_color(labels, num_labels), segmentada],
            ['Componentes conectados', 'Objetos segmentados'],
            'A4_componentes.png', 'A4. Segmentación final')

    # =========================================================================
    # B. CLASIFICACIÓN DE MONEDAS
    # =========================================================================
    print('\n=== B. Clasificación de monedas ===')
    objetos = medir_objetos(labels, num_labels)

    print(f"{'Obj':>3} {'Área':>8} {'Perím.':>8} {'rho':>6}  Tipo")
    monedas = [o for o in objetos if o['rho'] > RHO_MIN_MONEDA]
    dados = [o for o in objetos if o['rho'] <= RHO_MIN_MONEDA]
    centros, umbrales = clasificar_monedas(monedas)   # k-means sobre las áreas
    print(f"Centros de los grupos de área (k-means): {[round(float(c)) for c in centros]}")
    for o in objetos:
        tipo = MONEDAS[o['tipo']]['nombre'] if o in monedas else 'DADO'
        print(f"{o['id']:>3} {o['area']:>8.0f} {o['perimetro']:>8.0f} {o['rho']:>6.3f}  {tipo}")

    img_monedas = img.copy()
    for o in monedas:
        color = MONEDAS[o['tipo']]['color']
        cv2.drawContours(img_monedas, [o['cnt']], -1, color, 12)
        dibujar_texto(img_monedas, MONEDAS[o['tipo']]['nombre'].replace(' centavos', 'c'),
                      o['centro'], color)
    for o in dados:
        cv2.drawContours(img_monedas, [o['cnt']], -1, (255, 140, 0), 12)

    total = 0.0
    print('\nConteo de monedas:')
    for clave, m in MONEDAS.items():
        cant = sum(1 for o in monedas if o['tipo'] == clave)
        total += cant * m['valor']
        print(f"  {m['nombre']:<12}: {cant}")
    print(f'  Total monedas : {len(monedas)}   (valor total: ${total:.2f})')

    # Área vs. tipo para justificar los umbrales
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for clave, m in MONEDAS.items():
        areas = [o['area'] for o in monedas if o['tipo'] == clave]
        ax.scatter(areas, [m['nombre']] * len(areas), color=np.array(m['color']) / 255, s=80)
    for umbral in umbrales:   # corte entre grupos vecinos (punto medio de los centros)
        ax.axvline(umbral, color='gray', ls='--')
    ax.set_xlabel('Área (px)')
    ax.set_title('B1. Área de cada moneda y cortes de los grupos (k-means)')
    plt.tight_layout()
    fig.savefig(os.path.join(CARPETA_SALIDA, 'B1_areas.png'), dpi=150, bbox_inches='tight')
    plt.show(block=False)

    mostrar([img_monedas], ['B2. Monedas clasificadas por área'], 'B2_monedas.png')

    # =========================================================================
    # C. VALOR DE LOS DADOS
    # =========================================================================
    print('\n=== C. Valor de la cara superior de cada dado ===')
    img_dados = img.copy()
    valores = []
    for k, d in enumerate(sorted(dados, key=lambda o: o['centro'][0]), start=1):
        pips, descartados, oscuros, umbral = contar_pips(gris, d)
        valor = len(pips)
        valores.append(valor)
        print(f'Dado {k}: Otsu = {umbral:.0f} | candidatos = {len(pips) + len(descartados)} '
              f'| pips cara superior = {valor} | descartados = {len(descartados)}')
        for p in pips:
            print(f"    pip  : área = {p['area']:.0f}  rho = {p['rho']:.2f}")
        for p in descartados:
            print(f"    descartado: área = {p['area']:.0f}  rho = {p['rho']:.2f}")

        # Imagen de detalle del dado
        x, y, w, h = cv2.boundingRect(d['cnt'])
        pad = 40
        y0, y1, x0, x1 = max(y - pad, 0), y + h + pad, max(x - pad, 0), x + w + pad
        detalle = img[y0:y1, x0:x1].copy()
        for p in pips:
            cv2.drawContours(detalle, [p['cnt'] - [x0, y0]], -1, (0, 255, 0), 6)
        for p in descartados:
            cv2.drawContours(detalle, [p['cnt'] - [x0, y0]], -1, (255, 0, 0), 6)
        mostrar([img[y0:y1, x0:x1], oscuros[y0:y1, x0:x1], detalle],
                [f'Dado {k}', 'Píxeles oscuros dentro del dado',
                 'Verde: pips cara superior | Rojo: descartados'],
                f'C{k}_dado.png', f'C{k}. Dado {k}: valor = {valor}')

        cv2.drawContours(img_dados, [d['cnt']], -1, (255, 140, 0), 12)
        dibujar_texto(img_dados, str(valor), d['centro'], (255, 0, 0), escala=6, grosor=14)

    print(f'\nValores de los dados: {valores}  (suma = {sum(valores)})')
    mostrar([img_dados], [f'C. Valor de cada dado: {valores}'], 'C3_dados.png')

    # =========================================================================
    # Resultado final
    # =========================================================================
    img_final = img_monedas.copy()
    for k, d in enumerate(sorted(dados, key=lambda o: o['centro'][0])):
        dibujar_texto(img_final, str(valores[k]), d['centro'], (255, 0, 0), escala=6, grosor=14)
    mostrar([img_final], ['Resultado final: monedas clasificadas y valor de los dados'],
            'resultado_final.png')
    print(f'\nFiguras guardadas en: {CARPETA_SALIDA}')

    plt.show()


# =============================================================================
# Punto de entrada
# =============================================================================

if __name__ == '__main__':
    main()
