# Procesamiento de Imágenes — Trabajo Práctico N.º 2

Implementación en Python de técnicas de procesamiento de imágenes para la materia **Procesamiento de Imágenes — Licenciatura en Ciencia de Datos, Universidad Austral (2026)**.

El trabajo aborda dos problemas:

1. **Detección y clasificación de monedas y dados**
2. **Detección de patentes y segmentación de sus caracteres**

El enunciado está en `TUIA_PDI_TP2_2025_C2.pdf`. En ambos problemas los scripts informan por consola y muestran una figura en cada etapa del procesamiento.

---

## Problema 1 — Detección y clasificación de monedas y dados

Se procesa la imagen `images/monedas.jpg`, que contiene monedas de distinto valor y tamaño y dos dados sobre un fondo de intensidad no uniforme.

### A. Segmentación

1. Suavizado con un filtro gaussiano.
2. Pasaje al espacio **CIELAB**: canal L (brillo) y croma (intensidad del color). El croma ignora el degradé del fondo.
3. Binarización: croma alto (monedas doradas y bimetálicas) o brillo alto (monedas plateadas y dados). El umbral de brillo lo calcula **Otsu** usando solo los píxeles claros.
4. Operaciones morfológicas: clausura, relleno de huecos con `floodFill` y apertura.
5. Componentes conexas: cada objeto es una moneda o un dado.

### B. Clasificación y conteo de monedas

1. Para cada objeto se calcula contorno, área y perímetro.
2. Con ellos se obtiene el **factor de forma** `rho = 4·π·A / P²`: `rho > 0,8` es moneda y `rho` menor es dado.
3. Las monedas se clasifican por **área** con **k-means** de 3 grupos (10 centavos, 1 peso y 50 centavos), sin umbrales escritos a mano.

### C. Valor de la cara superior de cada dado

1. Se rellena la silueta del dado con su casco convexo.
2. Dentro del dado se buscan los puntos negros (pips) con Otsu, una apertura y una **clausura** que tapa los reflejos.
3. Un pip pertenece a la cara superior si es redondo (`rho > 0,65`) y su área es al menos la mitad del pip redondo más grande de ese dado. Los pips de las caras laterales se ven chicos y estirados.
4. El valor del dado es la cantidad de pips de la cara superior.

### Resultado sobre `monedas.jpg`

* 19 objetos detectados: 17 monedas y 2 dados.
* Monedas: 9 de 10 centavos, 5 de 1 peso y 3 de 50 centavos (total $7,40).
* Dados: 5 y 3.

---

## Problema 2 — Detección de patentes

Se procesan las doce imágenes `img01.png` a `img12.png` (carpeta `images/Patentes/`), que muestran la vista anterior o posterior de distintos vehículos con patentes argentinas del formato anterior: caja negra con 3 letras y 3 números claros.

### A. Detección y segmentación de la placa

1. Se descartan los márgenes izquierdo, derecho y superior de la foto, donde no hay patentes.
2. Se marcan como "negros" los píxeles con brillo bajo (canal V de HSV) y se construye un mapa de **saltos negro → claro**: cada trazo vertical de una letra deja un salto en su borde izquierdo.
3. Una apertura vertical elimina los saltos aislados (ruido).
4. Una ventana deslizante cuenta los saltos en cada posición; la que tiene más saltos es la patente. Se recentra el punto y se recorta la placa.

### B. Segmentación de los caracteres

1. Se pasa el recorte a grises, se estira el contraste y se binariza.
2. Con componentes conexas se obtienen las cajas candidatas, descartando las que no tienen proporciones de carácter.
3. Se eligen las cajas de **área parecida** (los seis caracteres miden casi lo mismo) y se igualan sus tamaños.
4. Si faltan caracteres, se deduce dónde están con la geometría de la patente: alineación, separación constante y un hueco entre letras y números.
5. **Enderezado:** se mide la inclinación de la patente, se gira el recorte y, con la misma matriz, las cajas. Los caracteres se recortan de la imagen girada.

Por cada foto se informa un estado:

| Estado         | Significado                                              |
| -------------- | -------------------------------------------------------- |
| `completo`     | se encontraron los 6 caracteres directamente             |
| `proyectado`   | faltaban caracteres y se dedujo su posición              |
| `insuficiente` | hay muy pocos caracteres para deducir los que faltan     |

### Resultado sobre las 12 fotos

* 9 completas (`img02`, `img04` a `img10` e `img12`) y 3 proyectadas (`img01`, `img03` e `img11`).
* Ninguna foto queda `insuficiente`.

### Limitaciones

* El código **ubica** los caracteres; no reconoce cuál es cada uno, porque la consigna solo pide segmentarlos.
* La proyección supone que falta como máximo un carácter por mitad.
* Si el cuerpo claro del auto se funde con el marco de la patente, las letras pueden quedar pegadas al marco.
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
problema_1.py                 Problema 1: monedas y dados
problema_2.py                 Problema 2: patentes (módulo con todo el algoritmo)
ejemplo_patente.py            Problema 2: paso a paso sobre una foto, un gráfico por paso
notebooks/ejemplo_patente.ipynb   El mismo paso a paso, como notebook
requirements.txt              Dependencias
TUIA_PDI_TP2_2025_C2.pdf      Enunciado
images/monedas.jpg            Imagen del Problema 1
images/Patentes/              Imágenes del Problema 2
images/output/                Figuras que genera problema_1.py
images/extra/                 Fotos adicionales para evaluar la robustez (usadas en el informe)
```

---

## Ejecución

Clonar el repositorio e instalar las dependencias:

```bash
git clone https://github.com/BMilesRouillon/tp_procesamiento_de_imagenes_2.git
cd tp_procesamiento_de_imagenes_2
pip install -r requirements.txt
```

Los scripts usan rutas relativas, por lo que se ejecutan **desde la carpeta del proyecto**.

**Problema 1** (tarda unos 30 segundos; guarda las figuras en `images/output/`):

```bash
python problema_1.py
```

**Problema 2** (informa por consola cada etapa y abre una figura por foto):

```bash
python problema_2.py
```

También se pueden indicar fotos puntuales:

```bash
python problema_2.py images/Patentes/img01.png images/Patentes/img05.png
```

**Paso a paso del Problema 2** sobre una foto:

```bash
python ejemplo_patente.py
```

o con el notebook `notebooks/ejemplo_patente.ipynb`, que se ejecuta con `notebooks/` como carpeta de trabajo (es lo que hace Jupyter por defecto) y por eso usa rutas que empiezan con `../`.

Al ejecutar `problema_1.py` se vuelven a generar los PNG de `images/output/`.
