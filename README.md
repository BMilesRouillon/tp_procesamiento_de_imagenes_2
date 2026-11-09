# Procesamiento de Imágenes — Trabajo Práctico N.º 2

Implementación en Python de técnicas de procesamiento de imágenes para la materia **Procesamiento de Imágenes I — Tecnicatura Universitaria en Inteligencia Artificial, UNR (2.º semestre 2025)**.

El trabajo aborda dos problemas:

1. **Detección y clasificación de monedas y dados**
2. **Detección de patentes y segmentación de sus caracteres**

---

## Problema 1 — Detección y clasificación de monedas y dados

Se desarrolla un algoritmo que procesa la imagen `monedas.jpg`, que contiene monedas de distinto valor y tamaño, y dados, sobre un fondo de intensidad no uniforme (se aclara hacia la derecha).

El script informa por consola y muestra una figura en cada etapa del procesamiento.

### A. Segmentación

1. Suavizado de la imagen con un filtro gaussiano.
2. Pasaje al espacio de color **CIELAB**: canal L (brillo) y croma (intensidad del color).
3. Binarización: croma alto (monedas doradas y bimetálicas) o brillo alto (monedas plateadas y dados). El croma ignora el degradé del fondo.
4. Operaciones morfológicas: clausura (tapa grietas), relleno de huecos con `floodFill` y apertura (elimina ruido).
5. Componentes conexas: cada isla blanca es una moneda o un dado.

### B. Clasificación y conteo de monedas

1. Para cada objeto se calcula contorno, área y perímetro.
2. Con ellos se obtiene el **factor de forma** `rho = 4·π·A / P²`, que vale 1 para un círculo perfecto. `rho` alto es moneda y `rho` bajo es dado.
3. Las monedas se clasifican por **área**, en tres grupos:

| Moneda      | Área aproximada (px) |
| ----------- | -------------------- |
| 10 centavos | 55 000 – 63 000      |
| 1 peso      | 84 000 – 87 000      |
| 50 centavos | ~105 000             |

### C. Valor de la cara superior de cada dado

1. Dentro de cada dado se buscan los puntos negros (pips) con un umbral de Otsu.
2. Los pips de la cara superior son círculos **grandes y redondos**; los de las caras laterales se ven chicos y estirados. Se filtran por área y por `rho`.
3. El valor del dado es la cantidad de pips de la cara superior.

### Resultado sobre `monedas.jpg`

* 19 objetos detectados: 17 monedas y 2 dados.
* Monedas: 9 de 10 centavos, 5 de 1 peso y 3 de 50 centavos (total $7,40).
* Dados: 5 y 3.

---

## Problema 2 — Detección de patentes

Se procesan las doce imágenes `img01.png` a `img12.png` (carpeta `images/Patentes/`), que muestran la vista anterior o posterior de distintos vehículos. Las patentes son del formato argentino anterior: caja negra con 3 letras y 3 números claros.

### A. Detección y segmentación de la placa

1. Se descartan los márgenes izquierdo, derecho y superior de la foto, donde no hay patentes.
2. Se marcan como "negros" los píxeles con brillo bajo (canal V de HSV).
3. Se construye un mapa de **saltos negro → claro**: cada trazo vertical de una letra deja un salto en su borde izquierdo.
4. Una apertura vertical elimina los saltos aislados (ruido) y conserva los bordes de las letras.
5. Una ventana deslizante del tamaño aproximado de una patente cuenta los saltos en cada posición. La ventana con más saltos es la patente.
6. Se recentra el punto con el promedio de las posiciones de los saltos y se recorta la placa.

### B. Segmentación de los caracteres

1. Se pasa el recorte a grises, se estira el contraste y se binariza.
2. Con componentes conexas (conectividad 4) se obtienen las cajas candidatas, descartando las que no tienen proporciones de carácter.
3. Se eligen las cajas de **área parecida** (los seis caracteres miden casi lo mismo) y se igualan sus tamaños.
4. Si faltan caracteres, se deduce dónde están con la geometría de la patente: inclinación entre caracteres, alineación y separación constante, con un hueco entre letras y números.

La función `analizar_foto` informa un estado por imagen:

| Estado         | Significado                                                       |
| -------------- | ----------------------------------------------------------------- |
| `completo`     | se encontraron los 6 caracteres directamente                      |
| `proyectado`   | faltaban caracteres y se dedujo su posición                       |
| `insuficiente` | hay muy pocos caracteres para deducir los que faltan              |

El detalle del método y de los parámetros ajustables está en `notebooks/README_patentes.md`.

### Limitaciones conocidas

* Si el cuerpo claro del auto se funde con el marco de la patente, las letras pueden quedar pegadas al marco.
* El "1" es más angosto y puede quedar fuera del grupo de cajas parecidas.
* La proyección supone que falta como máximo un carácter por mitad.
* El código **ubica** los caracteres; no reconoce cuál es cada uno, porque la consigna solo pide segmentarlos.
* Se ajustó con las 12 fotos disponibles; no se garantiza que generalice a otras condiciones de luz o ángulo.

---

## Tecnologías utilizadas

* **Python**
* **OpenCV**
* **NumPy**
* **Matplotlib**

---

## Estructura del repositorio

```text
problema_1.py                 Problema 1
notebooks/patentes.py         Problema 2: módulo con todo el algoritmo
notebooks/patentes.ipynb      Problema 2: notebook
notebooks/README_patentes.md  Problema 2: explicación paso a paso
notebooks/old/                Versiones de experimentación
images/monedas.jpg            Imagen del Problema 1
images/Patentes/              Imágenes del Problema 2
images/output/                Figuras del Problema 1
TUIA_PDI_TP2_2025_C2.pdf      Enunciado
```

---

## Ejecución

Clonar el repositorio:

```bash
git clone https://github.com/BMilesRouillon/tp_procesamiento_de_imagenes_2.git
cd tp_procesamiento_de_imagenes_2
```

Instalar las dependencias necesarias:

```bash
pip install opencv-python numpy matplotlib
```

**Problema 1.** Se ejecuta desde la carpeta del proyecto, porque usa rutas relativas (`images/monedas.jpg`). Las figuras se guardan en `images/output/`:

```bash
python problema_1.py
```

**Problema 2.** Se pasan las rutas de las fotos como argumentos y se abre una figura por foto:

```bash
python notebooks/patentes.py images/Patentes/img01.png images/Patentes/img02.png
```

También se puede usar el notebook `notebooks/patentes.ipynb`.

---

## Resultados

El proyecto permite automatizar el procesamiento de las imágenes del trabajo práctico, incluyendo:

* segmentación automática de monedas y dados sobre un fondo no uniforme;
* clasificación y conteo de monedas por tipo;
* lectura del valor de la cara superior de cada dado;
* localización automática de la patente en la foto;
* segmentación de los seis caracteres de la patente, deduciendo por geometría los que no se detectan directamente;
* figuras con los resultados de cada etapa.

El desarrollo se realizó con operaciones básicas de procesamiento de imágenes, sin redes neuronales ni librerías de lectura de patentes.
