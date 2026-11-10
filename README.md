# Procesamiento de Imágenes — Trabajo Práctico N.º 2

Implementación en Python de técnicas de procesamiento de imágenes para la materia **Procesamiento de Imágenes — Licenciatura en Ciencia de Datos, Universidad Austral (2026)**.

El trabajo aborda dos problemas:

1. **Detección y clasificación de monedas y dados**
2. **Detección de patentes y segmentación de sus caracteres**

Cada script informa por consola y muestra una figura en cada etapa del procesamiento.

---

## Problema 1 — Detección y clasificación de monedas y dados

Se procesa la imagen `monedas.jpg`, que contiene monedas de distinto valor y tamaño y dos dados, sobre un fondo de intensidad no uniforme (se aclara hacia la derecha).

### A. Segmentación

1. Suavizado de la imagen con un filtro gaussiano.
2. Pasaje al espacio de color **CIELAB**: canal L (brillo) y croma (intensidad del color).
3. Binarización: croma alto (monedas doradas y bimetálicas) o brillo alto (monedas plateadas y dados). Los dos umbrales se **miden en el borde de la foto**, donde solo hay fondo: un píxel es objeto si tiene más color o más brillo que cualquier punto del fondo. El croma ignora el degradé del fondo.
4. Operaciones morfológicas: clausura (tapa grietas), relleno de huecos con `floodFill` y apertura (elimina ruido).
5. Componentes conexas: cada isla blanca es una moneda o un dado.

### B. Clasificación y conteo de monedas

1. Para cada objeto se calcula contorno, área y perímetro.
2. Con ellos se obtiene el **factor de forma** `rho = 4·π·A / P²`, que vale 1 para un círculo perfecto. `rho` alto es moneda y `rho` bajo es dado.
3. Las monedas se agrupan por **área** con **k-means** en tres grupos (10 centavos, 1 peso y 50 centavos), sin umbrales de área escritos a mano.

### C. Valor de la cara superior de cada dado

1. Dentro de cada dado (rellenado con su casco convexo) se buscan los puntos negros con un umbral de Otsu, y una clausura tapa los reflejos.
2. Los puntos de la cara superior son **grandes y redondos**; los de las caras laterales se ven chicos y estirados. Se filtran por `rho` y por área relativa al punto más grande del mismo dado.
3. El valor del dado es la cantidad de puntos de la cara superior.

### Resultado sobre `monedas.jpg`

* 19 objetos detectados: 17 monedas y 2 dados.
* Monedas: 9 de 10 centavos, 5 de 1 peso y 3 de 50 centavos (total $7,40).
* Dados: 5 y 3.

---

## Problema 2 — Detección de patentes

Se procesan las doce imágenes `img01.png` a `img12.png` (carpeta `images/Patentes/`). Las patentes son del formato argentino anterior: caja negra con 3 letras y 3 números claros.

### A. Detección y segmentación de la placa

1. Se descartan los márgenes izquierdo, derecho y superior de la foto, donde no hay patentes.
2. Se marcan como "negros" los píxeles con brillo bajo (canal V de HSV).
3. Se construye un mapa de **saltos negro → claro**: cada trazo vertical de una letra deja un salto en su borde izquierdo.
4. Una erosión vertical elimina los saltos aislados (ruido) y una dilatación devuelve el grosor a los bordes de las letras.
5. Una ventana deslizante del tamaño aproximado de una patente cuenta los saltos en cada posición. La ventana con más saltos es la patente.
6. Se recentra el punto con el promedio de las posiciones de los saltos y se recorta la zona.
7. Con los caracteres ya ubicados y la patente enderezada (punto B), se **segmenta la placa**: partiendo de la fila de caracteres, el rectángulo se agranda hacia cada lado mientras la columna o fila siga siendo mayormente oscura, hasta llegar al marco claro.

### B. Segmentación de los caracteres

1. Se pasa el recorte a grises, se estira el contraste y se binariza.
2. Con componentes conexas (conectividad 4) se obtienen las cajas candidatas, descartando las que no tienen proporciones de carácter.
3. Se eligen las cajas de **área parecida** (los seis caracteres miden casi lo mismo) y se igualan sus tamaños.
4. Si faltan caracteres, se deduce dónde están con la geometría de la patente: inclinación entre caracteres, alineación y separación constante, con un hueco entre letras y números.
5. Se mide la inclinación de la fila, se gira el recorte para enderezarlo y se recorta cada carácter.

La función `analizar_foto` informa un estado por imagen:

| Estado         | Significado                                                       |
| -------------- | ----------------------------------------------------------------- |
| `completo`     | se encontraron los 6 caracteres directamente                      |
| `proyectado`   | faltaban caracteres y se dedujo su posición                       |
| `insuficiente` | hay muy pocos caracteres para deducir los que faltan              |

Resultado: en las 12 fotos se segmenta la placa y se obtienen los 6 caracteres (9 `completo` y 3 `proyectado`: img01, img03 e img11).

### Limitaciones conocidas

* El "1" y la "M" tienen un ancho distinto al resto; las cajas igualadas pueden cortarlos un poco.
* Con pocos píxeles por carácter (img11) las cajas quedan algo más chicas que el carácter.
* La proyección supone que falta como máximo un carácter por mitad.
* El código **ubica** los caracteres; no reconoce cuál es cada uno, porque la consigna solo pide segmentarlos.
* Se ajustó con las 12 fotos disponibles; no se garantiza que generalice a otras condiciones de luz o ángulo.

---

## Estructura del repositorio

```text
problema_1.py                   Problema 1
problema_2.py                   Problema 2: todo el algoritmo y la visualización
ejemplo_patente.py              Problema 2: paso a paso de una foto (img01), una figura por etapa
notebooks/ejemplo_patente.ipynb El mismo paso a paso en notebook
images/monedas.jpg              Imagen del Problema 1
images/Patentes/                Imágenes del Problema 2
images/output/                  Figuras del Problema 1
TUIA_PDI_TP2_2025_C2.pdf        Enunciado
```

---

## Ejecución

```bash
pip install opencv-python numpy matplotlib
```

Los scripts usan rutas relativas: se ejecutan **desde la carpeta del proyecto**.

```bash
python problema_1.py
```

```bash
python problema_2.py
```

Sin argumentos procesa las 12 fotos (una figura por foto). También se pueden pasar fotos sueltas:

```bash
python problema_2.py images/Patentes/img01.png images/Patentes/img05.png
```

```bash
python ejemplo_patente.py
```

---

## Tecnologías utilizadas

* **Python**, **OpenCV**, **NumPy** y **Matplotlib**.
* Sin redes neuronales ni librerías de lectura de patentes.
