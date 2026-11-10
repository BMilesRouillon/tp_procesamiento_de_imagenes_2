"""Funciones compartidas por los notebooks de notebooks/benja/.
Reusa el patentes.py original (sin cambios) y le suma rotación, binarización múltiple,
reconocimiento por huellas y verificación por contornos."""

import csv
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))        # patentes.py (original)
import patentes as P                                # noqa: E402

RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
RUTA_AUX = os.path.join(RAIZ, "images", "auxiliares")
FOTOS = range(1, 13)

# Texto real de cada patente de la consigna (escrito a mano mirando las fotos)
ETIQUETAS = {1: "BUF817", 2: "GXQ798", 3: "DMH024", 4: "LHE986", 5: "FJY344", 6: "LEH992",
             7: "LOD123", 8: "JMO337", 9: "FFD765", 10: "DOA322", 11: "GMQ569", 12: "BQX387"}

AN, AL = 8, 12              # tamaño de la huella de cada carácter
UMBRAL_SOLIDEZ = 0.905      # por debajo: forma con un lado recto (D); por encima: redonda (O)
DIF_ALTURA_AGUJERO = 0.12   # diferencia máxima de altura para considerar "el mismo" agujero


def ruta_foto(i):
    return os.path.join(RAIZ, "images", "Patentes", f"img{i:02d}.png")


# ---------- Rotación ----------

def binaria_tophat(gris, k=7):
    """Top-hat: deja solo los trazos claros y finos (las letras). Después, Otsu."""
    th = cv2.morphologyEx(gris, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def rotar(img, angulo):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def estimar_angulo(recorte, rango=20):
    """Ángulo entre -20° y +20° que deja más marcadas las filas de tinta."""
    b = binaria_tophat(cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY))
    angulos = np.arange(-rango, rango + 0.1, 1.0)
    return max(angulos, key=lambda a: rotar(b, a).sum(axis=1).astype(float).var())


# ---------- Binarización múltiple y segmentación ----------

def variantes(rotada):
    """32 versiones binarias del recorte: 4 fuentes de gris x 8 binarizaciones."""
    gris = cv2.cvtColor(rotada, cv2.COLOR_BGR2GRAY)
    fuentes = {"gris": gris,
               "L": cv2.cvtColor(rotada, cv2.COLOR_BGR2LAB)[..., 0],
               "V": cv2.cvtColor(rotada, cv2.COLOR_BGR2HSV)[..., 2],
               "eq": cv2.equalizeHist(gris)}
    salida = {}
    for nombre, fuente in fuentes.items():
        est = cv2.normalize(fuente, None, 0, 255, cv2.NORM_MINMAX)
        for u in (110, 127, 150):
            salida[f"{nombre}/umbral{u}"] = np.where(est > u, 255, 0).astype(np.uint8)
        salida[f"{nombre}/otsu"] = cv2.threshold(est, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        for k in (3, 5, 7, 9):
            salida[f"{nombre}/tophat{k}"] = binaria_tophat(est, k)
    return salida


def cajas_de_variante(binaria):
    """Método original: componentes conexas, cajas de área parecida, cajas iguales."""
    return P.igualar_cajas(P.cajas_de_area_similar(P.componentes(binaria)))


def segmentar(rotada):
    """Elige la versión más parecida a la mediana entre las que encuentran 6 cajas.
    Si ninguna llega a 6, completa con la geometría del patentes.py original."""
    vs = variantes(rotada)
    cajas = {n: cajas_de_variante(b) for n, b in vs.items()}
    validas = {n: c for n, c in cajas.items() if len(c) == 6}
    if validas:
        mediana = np.median([[c.cx for c in cs] for cs in validas.values()], axis=0)
        costo = lambda cs: sum(abs(c.cx - m) for c, m in zip(cs, mediana)) + np.std([c.cy for c in cs])
        nombre = min(validas, key=lambda n: costo(validas[n]))
        return dict(version=nombre, binaria=vs[nombre], cajas=validas[nombre],
                    metodo="consenso", validas=len(validas), total=len(vs))
    nombre = max(cajas, key=lambda n: len(cajas[n]))
    todas, metodo = cajas[nombre], "insuficiente"
    if len(todas) >= 2:
        _, proyectadas = P.completar_caracteres(todas, vs[nombre].shape[1])
        extra = [P.Caja(int(round(p.x0)), int(round(p.y0)), int(round(p.x1 - p.x0)), int(round(p.y1 - p.y0)))
                 for p in proyectadas]
        if extra:
            todas, metodo = sorted(todas + extra, key=lambda c: c.cx), "geometria"
    return dict(version=nombre, binaria=vs[nombre], cajas=todas, metodo=metodo, validas=0, total=len(vs))


# ---------- Caracteres y huellas ----------

def recortar_caracter(binaria, caja):
    """Recorta la caja y la ajusta al rectángulo mínimo de tinta."""
    H, W = binaria.shape
    sub = binaria[max(0, caja.y):min(H, caja.y + caja.h), max(0, caja.x):min(W, caja.x + caja.w)]
    puntos = cv2.findNonZero(sub)
    if puntos is None:
        return sub
    x, y, w, h = cv2.boundingRect(puntos)
    return sub[y:y + h, x:x + w]


def huella(caracter):
    """Mini-imagen de 8x12 (0 = fondo, 1 = tinta) aplanada en un vector."""
    chica = cv2.resize(caracter, (AN, AL), interpolation=cv2.INTER_AREA)
    return (chica.astype(np.float32) / 255).flatten()


def procesar_recorte(recorte):
    """Endereza, segmenta y calcula las huellas de un recorte de patente."""
    angulo = estimar_angulo(recorte)
    rotada = rotar(recorte, angulo)
    r = dict(recorte=recorte, angulo=angulo, rotada=rotada, **segmentar(rotada))
    r["caracteres"] = [recortar_caracter(r["binaria"], c) for c in r["cajas"]]
    r["huellas"] = [huella(ch) for ch in r["caracteres"]]
    return r


def cargar_tp():
    """Procesa las 12 fotos de la consigna: {número de foto: resultado}."""
    return {i: procesar_recorte(P.preparar_foto(ruta_foto(i))[0]) for i in FOTOS}


# ---------- Patentes auxiliares ----------

def leer_auxiliares():
    """Filas de images/auxiliares/auxiliares.csv."""
    with open(os.path.join(RUTA_AUX, "auxiliares.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def preparar_auxiliar(fila, ancho=102, alto=50):
    """Recorta si la fila lo indica y reduce a 102 x 50 px (tamaño de los recortes del TP)."""
    img = cv2.imread(os.path.join(RUTA_AUX, fila["archivo"]))
    if fila["recorte"]:
        x0, y0, x1, y1 = [int(v) for v in fila["recorte"].split(",")]
        img = img[y0:y1, x0:x1]
    h, w = img.shape[:2]
    esc = min(ancho / w, alto / h)
    chica = cv2.resize(img, (max(1, int(round(w * esc))), max(1, int(round(h * esc)))), interpolation=cv2.INTER_AREA)
    dh, dw = alto - chica.shape[0], ancho - chica.shape[1]
    return cv2.copyMakeBorder(chica, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2, cv2.BORDER_REPLICATE)


def cargar_auxiliares():
    """Procesa las auxiliares marcadas con usar_en_modelo = si."""
    salida = []
    for fila in leer_auxiliares():
        if fila["usar_en_modelo"] == "si":
            r = procesar_recorte(preparar_auxiliar(fila))
            salida.append(dict(r, texto=fila["texto"], archivo=fila["archivo"]))
    return salida


# ---------- Variantes artificiales y contornos ----------

def aumentar_caracter(caracter):
    """9 versiones del carácter: giros de -4, 0 y +4 grados x anchos de 0.9, 1.0 y 1.1."""
    base = cv2.copyMakeBorder(caracter, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=0)
    H, W = base.shape
    salida = []
    for ang in (-4, 0, 4):
        for sx in (0.9, 1.0, 1.1):
            Mx = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
            Mx[0, :2] *= sx
            Mx[0, 2] = W / 2 - (Mx[0, 0] * W / 2 + Mx[0, 1] * H / 2)
            r = (cv2.warpAffine(base, Mx, (W, H), flags=cv2.INTER_LINEAR) > 100).astype(np.uint8) * 255
            puntos = cv2.findNonZero(r)
            if puntos is not None:
                x, y, w, h = cv2.boundingRect(puntos)
                salida.append(r[y:y + h, x:x + w])
    return salida


def firma_contornos(caracter, escala=6):
    """Cantidad de agujeros, su altura (0 = arriba, 1 = abajo) y solidez (área / casco convexo)."""
    g = cv2.resize(caracter, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    g = cv2.threshold(cv2.copyMakeBorder(g, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0), 127, 255, cv2.THRESH_BINARY)[1]
    H, W = g.shape
    contornos, jerarquia = cv2.findContours(g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    alturas, externo = [], None
    for c, j in zip(contornos, jerarquia[0] if jerarquia is not None else []):
        if j[3] != -1:                                       # contorno dentro de otro = agujero
            m = cv2.moments(c)
            if cv2.contourArea(c) >= 0.01 * H * W and m["m00"] > 0:
                alturas.append(m["m01"] / m["m00"] / H)
        elif externo is None or cv2.contourArea(c) > cv2.contourArea(externo):
            externo = c
    solidez = 1.0
    if externo is not None and cv2.contourArea(cv2.convexHull(externo)) > 0:
        solidez = cv2.contourArea(externo) / cv2.contourArea(cv2.convexHull(externo))
    return dict(n=len(alturas), y=float(np.mean(alturas)) if alturas else None, solidez=solidez)


def desajuste(a, b):
    """Reglas de contornos que no se cumplen entre dos firmas (0 = compatibles).
    Distinta cantidad de agujeros: 1. Con un agujero cada uno: otra altura y/o distinta solidez, 1 cada una."""
    if a["n"] != b["n"]:
        return 1
    if a["n"] != 1:
        return 0
    return (abs(a["y"] - b["y"]) > DIF_ALTURA_AGUJERO) + ((a["solidez"] < UMBRAL_SOLIDEZ) != (b["solidez"] < UMBRAL_SOLIDEZ))


# ---------- Reconocimiento ----------

def armar_referencias(resultados_tp, auxiliares=(), aumentar=False):
    """Lista de (huella, carácter, es_letra, origen, firma). aumentar=True suma 9 variantes por carácter."""
    refs = []
    fuentes = [(r, ETIQUETAS[i], f"img{i:02d}") for i, r in resultados_tp.items()]
    fuentes += [(a, a["texto"], a["archivo"]) for a in auxiliares]
    for r, texto, origen in fuentes:
        if len(r["huellas"]) != 6:
            continue
        for p in range(6):
            for ch in (aumentar_caracter(r["caracteres"][p]) if aumentar else [r["caracteres"][p]]):
                refs.append((huella(ch), texto[p], p < 3, origen, firma_contornos(ch)))
    return refs


def reconocer_caracter(h, firma, es_letra, refs, excluir=None, peso=0.0):
    """Vecino más cercano del mismo tipo. distancia = huellas + peso x desajuste de contornos."""
    candidatos = [(np.linalg.norm(h - rh) + peso * desajuste(firma, rf), c)
                  for rh, c, letra, origen, rf in refs if letra == es_letra and origen != excluir]
    return min(candidatos)[1]


def evaluar(resultados_tp, refs, peso=0.0):
    """Lee cada patente de la consigna dejándola afuera de las referencias.
    Devuelve ({foto: texto leído}, resumen)."""
    lecturas = {}
    for i, r in resultados_tp.items():
        if len(r["huellas"]) != 6:
            lecturas[i] = "??????"
            continue
        lecturas[i] = "".join(
            reconocer_caracter(r["huellas"][p], firma_contornos(r["caracteres"][p]), p < 3, refs, f"img{i:02d}", peso)
            for p in range(6))
    bien = lambda posiciones: sum(lecturas[i][p] == ETIQUETAS[i][p] for i in lecturas for p in posiciones)
    resumen = dict(letras=bien(range(3)), numeros=bien(range(3, 6)),
                   completas=sum(lecturas[i] == ETIQUETAS[i] for i in lecturas),
                   aciertos={i: sum(a == b for a, b in zip(lecturas[i], ETIQUETAS[i])) for i in lecturas})
    return lecturas, resumen


# ---------- Resultados por patente ----------

def guardar_figura_patente(i, resultados_tp, base, lectura, carpeta, nota=""):
    """Guarda patente_XX.png: foto original, método original (antes), modelo nuevo (después) y cada carácter leído."""
    r, b = resultados_tp[i], base[i]
    img = cv2.imread(ruta_foto(i))
    h, w = img.shape[:2]
    cx, cy, _, _ = P.detectar_punto(img)
    cw, ch = int(P.RECORTE_W * w), int(P.RECORTE_H * h)
    x0, y0 = min(max(0, cx - cw // 2), w - cw), min(max(0, cy - ch // 2), h - ch)

    fig = plt.figure(figsize=(13, 7))
    g = fig.add_gridspec(2, 6, height_ratios=[1.5, 1])
    a = fig.add_subplot(g[0, 0:2])
    a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    a.add_patch(plt.Rectangle((x0, y0), cw, ch, fill=False, edgecolor="lime", linewidth=2))
    a.set_title("foto original y recorte")

    a = fig.add_subplot(g[0, 2:4])
    a.imshow(cv2.cvtColor(b.recorte, cv2.COLOR_BGR2RGB))
    for xa, ya, xb, yb, tipo in b.cajas_finales():
        a.add_patch(plt.Rectangle((xa, ya), xb - xa, yb - ya, fill=False, linewidth=1.5,
                                  edgecolor="lime" if tipo == "detectada" else "orange"))
    a.set_title(f"Antes (original): {b.estado}, {len(b.encontradas)} detectadas")

    a = fig.add_subplot(g[0, 4:6])
    a.imshow(cv2.cvtColor(r["rotada"], cv2.COLOR_BGR2RGB))
    for c in r["cajas"]:
        a.add_patch(plt.Rectangle((c.x, c.y), c.w, c.h, fill=False, edgecolor="lime", linewidth=1.5))
    a.set_title(f"Después: giro {r['angulo']:+.0f}°, {r['metodo']}\n{r['version']}", fontsize=10)

    gris = cv2.cvtColor(r["rotada"], cv2.COLOR_BGR2GRAY)
    for p, c in enumerate(r["cajas"][:6]):
        e = fig.add_subplot(g[1, p])
        e.imshow(gris[max(0, c.y):c.y + c.h, max(0, c.x):c.x + c.w], cmap="gray")
        e.set_title(f"leído {lectura[p]}  (real {ETIQUETAS[i][p]})",
                    color="green" if lectura[p] == ETIQUETAS[i][p] else "red", fontsize=11)
    for e in fig.axes:
        e.axis("off")

    ok = sum(x == y for x, y in zip(lectura, ETIQUETAS[i]))
    fig.suptitle(f"img{i:02d}   leída: {lectura[:3]} {lectura[3:]}   |   real: {ETIQUETAS[i][:3]} {ETIQUETAS[i][3:]}"
                 f"   |   {ok}/6 bien{nota}", fontsize=14)
    fig.tight_layout()
    os.makedirs(carpeta, exist_ok=True)
    fig.savefig(os.path.join(carpeta, f"patente_{i:02d}.png"), dpi=110)
    return fig


def guardar_resultados(carpeta, resultados_tp, base, lecturas, nota=""):
    """Guarda las 12 figuras y resumen.csv en la carpeta."""
    for i in FOTOS:
        plt.close(guardar_figura_patente(i, resultados_tp, base, lecturas[i], carpeta, nota))
    with open(os.path.join(carpeta, "resumen.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["foto", "estado_original", "angulo", "metodo", "version", "cajas", "leida", "real", "aciertos"])
        for i, r in resultados_tp.items():
            w.writerow([f"img{i:02d}", base[i].estado, f"{r['angulo']:+.0f}", r["metodo"], r["version"], len(r["cajas"]),
                        lecturas[i], ETIQUETAS[i], sum(x == y for x, y in zip(lecturas[i], ETIQUETAS[i]))])
