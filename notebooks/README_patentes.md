# Detección de caracteres en patentes argentinas — cómo funciona, paso a paso

Este documento explica, sin asumir conocimientos previos de procesamiento de imágenes,
qué hace `patentes.py` y por qué. El código está en ese archivo; acá está la idea.

---

## 0. El problema

Tenemos fotos de autos de 640×360 píxeles. En cada una hay una patente argentina del
formato viejo: **caja negra con caracteres claros, 3 letras + espacio + 3 números**
(por ejemplo `BUF 817`). Queremos:

1. Encontrar **dónde** está la patente en la foto.
2. Encontrar **dónde está cada uno de los 6 caracteres** (su caja contenedora), para
   poder después reconocerlos uno por uno.

No usamos redes neuronales ni librerías de lectura de patentes: solo operaciones básicas
de imágenes y un poco de geometría. Priorizamos que sea simple y entendible, aunque
no funcione en el 100% de los casos.

## 1. Una imagen es una matriz de números

Una foto en color es una matriz de alto × ancho × 3: para cada píxel, tres números de 0 a 255
(azul, verde y rojo en OpenCV). Una imagen en grises es una matriz de alto × ancho, donde
0 = negro y 255 = blanco. **Todo el algoritmo es aritmética sobre esas matrices.**

Convención: `x` crece hacia la derecha e `y` crece hacia **abajo**.

## 2. Etapa 1 — Encontrar la patente (`detectar_punto`)

**Idea:** una patente es una caja negra con trazos claros verticales (las letras). Si
recorremos una fila de izquierda a derecha, cada vez que pasamos de *negro* a *claro*
estamos en el borde izquierdo de un trazo. Una zona con **muchos** de esos bordes juntos
casi seguro es una patente.

Paso a paso:

1. **Descartar márgenes.** Se ignora el 20% izquierdo, el 20% derecho y el 20% superior de la foto:
   ahí nunca hay patentes (cielo, bordes). Esto reduce los falsos positivos.
2. **Píxeles negros.** Se convierte la foto a HSV y se usa el canal **V** (brillo, 0–255).
   Un píxel es "negro" si `V < 80`. No se mira el color: un auto rojo oscuro tiene el
   color intenso pero se distingue de la caja de la patente porque su brillo no es tan bajo.
3. **Mapa de saltos.** Para cada píxel se pregunta: *¿el píxel de su izquierda es negro y
   él no?* Si sí, vale 1; si no, 0. (`mapa_transiciones`)
   ```
   fila de píxeles:   N N N C C N N C C C      (N = negro, C = claro)
   saltos:            0 0 0 1 0 0 0 1 0 0
   ```
4. **Limpieza (apertura vertical).** El ruido produce saltos sueltos; los bordes de letras
   forman **líneas verticales** de varios píxeles. Entonces:
   - **Erosión** con un kernel vertical de 3×1: un salto sobrevive solo si tiene saltos
     arriba y abajo. Los puntos aislados desaparecen.
   - **Dilatación** con 3×1: los trazos que sobrevivieron se vuelven a engrosar.
   Erosión seguida de dilatación se llama **apertura**.
5. **¿Dónde hay más saltos?** Se deja deslizar una ventana del tamaño aproximado de una
   patente (11% del ancho × 9% del alto de la foto) y en cada posición se calcula
   qué fracción de la ventana son saltos (un *promedio móvil*, `cv2.blur`). La posición
   con el valor más alto (`argmax`) es el candidato. Fuera de la imagen se cuenta como 0
   para que los bordes no ganen por error.
6. **Recentrar.** La ventana ganadora puede estar un poco corrida. Se toma una zona de
   2W×2H alrededor, se promedian las posiciones de todos los saltos y ese promedio es el punto final,
   que cae en medio de los caracteres.

El resultado es **un punto dentro de la patente**.

## 3. Recorte (`recortar`)

Se recorta un rectángulo de 16% del ancho × 14% del alto de la foto (≈102×50 px) centrado
en ese punto. Si el punto está cerca del borde, el rectángulo se **desplaza** en vez de achicarse,
así todos los recortes tienen el mismo tamaño. A partir de acá trabajamos **solo con el recorte**.

## 4. Etapa 2 — Blanco y negro (`preparar_binaria`)

1. **A grises.**
2. **Estirar el contraste:** el píxel más oscuro pasa a 0, el más claro a 255 y todo lo demás
   proporcionalmente. Sirve para que fotos oscuras o claras queden en la misma escala.
3. **Binarizar:** si el valor es mayor a 127 → blanco (255); si no → negro (0).

Queda una imagen de solo dos valores donde, idealmente, **los caracteres son blancos
separados entre sí**.

## 5. Candidatos a caracteres (`componentes`)

Una **componente conexa** es un grupo de píxeles blancos pegados entre sí. OpenCV las
encuentra todas (`connectedComponentsWithStats`) y para cada una da su caja contenedora.
Se usa conectividad **4**: dos píxeles son vecinos solo si comparten un lado (no cuenta
la diagonal). Eso evita pegar un carácter con otro por un único píxel en diagonal.

No toda componente es un carácter. Se descartan las que:
- tienen menos de 6 píxeles (ruido),
- miden menos del 20% o más del 70% del alto del recorte,
- son más anchas que altas (los caracteres de una patente son más altos que anchos).

## 6. Elegir los que se parecen (`cajas_de_area_similar`)

Los 6 caracteres tienen cajas de **tamaño casi igual**, mientras que el ruido y el marco
de la patente tienen tamaños variados. Entonces:

1. Se calcula el área de cada caja (`ancho × alto`).
2. Cada caja "vota" por todas las que tienen un área parecida a la suya (diferencia menor al 25%).
3. Gana el área con más votos y se queda el grupo de cajas parecidas a ella.
   Si hay más de 6, se dejan las 6 más cercanas al área ganadora.

### 6.1 Igualar las cajas (`igualar_cajas`)

A veces una caja no abarca todo el carácter (queda recortada). Como todos los caracteres miden
lo mismo, se toma la caja **más grande** y se reconstruye cada caja con ese ancho y alto,
**centrada en el centro de la caja original**. Desde acá, todo se calcula con estas cajas iguales.

**Si quedaron exactamente 6, terminamos.** Si quedaron menos (porque una letra se pegó al
marco, por ejemplo), pasamos a la etapa 3.

## 7. Etapa 3 — Deducir los que faltan con geometría

Sabemos cómo es una patente: los 6 caracteres están alineados sobre una recta, **separados
por una distancia constante**, con un hueco entre las letras y los números. Usamos eso.

### 7.1 La inclinación (`pendiente_entre`)

Si hay al menos 2 caracteres, se traza una recta entre los **centros de las cajas de la primera y la
última** (las más separadas, así la inclinación se estima mejor). Su pendiente `dy/dx` es la
inclinación de la patente en la foto. Un valor de −0,1 equivale a unos −6°.

### 7.2 El "techo" y el "piso" (`altura_techo`, `altura_piso`)

Con esa misma pendiente se trazan dos rectas más sobre la primera caja encontrada:
- una por su **esquina superior izquierda** (techo),
- otra por su **esquina inferior derecha** (piso).

Los demás caracteres de la patente quedan entre ambas rectas (en nuestras pruebas, sus esquinas
caían a menos de 2 px). Esa franja nos dice **a qué altura** va cualquier carácter faltante.

### 7.3 La línea del 50% y las dos mitades (`proyectar_lado`)

El recorte está centrado aproximadamente en la patente, así que su línea vertical del 50% cae
cerca del hueco entre las letras y los números. A cada lado debería haber **3** caracteres.

Si en una mitad hay exactamente **2**, hay que decidir dónde falta el tercero. Las dos
mitades se tratan igual midiendo distancias "hacia afuera" desde la línea. Sean *A* la caja más
cercana a la línea y *B* la más lejana, con `paso` = distancia entre sus centros:

| Situación | Conclusión |
|---|---|
| El espacio libre entre A y B es **mayor que el ancho de una caja** | falta uno **en el medio** (punto medio entre A y B) |
| A y B son vecinas y A está **muy lejos de la línea** (más de 1,5 pasos): entra otro | falta el que está **hacia el centro** (a un paso de A, hacia la línea) |
| A y B son vecinas y A está cerca de la línea | falta el **último, en el extremo** (a un paso de B, hacia afuera) |

Por qué 1,5: en la patente, el primer carácter de una mitad está a 1 paso de la línea del
50% y el segundo a 2 pasos. El 1,5 es el punto medio entre ambos casos, para tolerar ruido.

**Cuidado:** la línea del 50% del recorte no coincide exactamente con el centro de la patente (el
recorte se centra en el promedio de los bordes verticales, no en la patente). En nuestras pruebas la
línea quedaba pegada al último carácter de la izquierda y a 1–1,5 anchos del primero de la derecha, por
lo que el caso "falta el último de la derecha" suele confundirse con "falta el del centro".

### 7.4 La caja proyectada

La caja que se dibuja es un rectángulo recto, como las detectadas:
- su **ancho** es el promedio de A y B,
- su **esquina superior izquierda** se apoya en la recta del techo,
- su **esquina inferior derecha** se apoya en la recta del piso.

## 8. Resultado de `analizar_foto`

Devuelve un objeto con el recorte, la binaria, las candidatas, las cajas encontradas,
la pendiente y las cajas proyectadas, y un `estado`:

- `completo`: se encontraron los 6 directamente.
- `proyectado`: faltaban y se dedujo dónde están.
- `insuficiente`: muy pocos caracteres para deducir (menos de 2, o mitades con 0, 1 o más de 3).

## 9. Cómo usarlo

```python
from patentes import analizar_foto, ver_resultados

r = analizar_foto("../images/Patentes/img03.png")
print(r.estado, r.encontradas, r.proyectadas)

ver_resultados([f"../images/Patentes/img{i:02d}.png" for i in range(1, 13)])
```

Cada foto abre su propia figura: arriba el recorte, la binaria con las candidatas y el resultado; abajo cada carácter por
separado (grises y binario) con su tipo y sus valores (`x`, `y`, ancho×alto).

Colores del gráfico: **celeste** = candidatas, **amarillo** = cajas originales, **verde** = cajas igualadas, **naranja** = deducidas,
línea **blanca** vertical = 50% del ancho, **roja** = recta por los centros,
**celeste/magenta** = techo/piso.

## 10. Parámetros que se pueden ajustar (al inicio de `patentes.py`)

| Parámetro | Qué controla | Si lo subís… |
|---|---|---|
| `V_NEGRO` | brillo máximo para considerar un píxel negro | se considera negro más píxeles |
| `ALTO_MIN_TRAZO` | alto mínimo de un borde vertical | se elimina más ruido (y trazos cortos) |
| `VENTANA_W/H` | tamaño de la ventana de búsqueda | se busca un objeto más grande |
| `RECORTE_W/H` | tamaño del recorte | el recorte incluye más contexto |
| `UMBRAL_BINARIO` | umbral blanco/negro tras estirar contraste | menos píxeles quedan blancos |
| `CONECTIVIDAD` | 4 u 8 vecinos para componentes | 8 pega más píxeles en diagonal |
| `TOL_AREA` | tolerancia de área entre cajas parecidas | entran cajas más distintas (p. ej. el "1", angosto) |
| `MARGEN_PROYECCION` | cuántos pasos de la línea hacen falta para decidir "hacia el centro" | se proyecta más seguido "en el extremo" |

## 11. Limitaciones conocidas

- Si el cuerpo claro del auto se funde con el marco de la patente en la binarización, las letras
  pueden quedar pegadas al marco y no detectarse como componentes separadas.
- El "1" es más angosto: su área de caja es menor y puede quedar fuera del grupo (`TOL_AREA`).
- Que haya exactamente 6 cajas de área parecida no **garantiza** que sean los 6 caracteres reales (una pieza
  del marco podría colarse).
- La proyección asume que la línea del 50% del recorte cae cerca del hueco entre letras y números, pero
  suele estar corrida hacia la izquierda; y que falta un solo carácter por mitad.
- Este código **ubica** los caracteres; todavía **no los reconoce** (leer cuál letra o número es cada uno).
- Se desarrolló y ajustó con 12 fotos; no se garantiza que generalice a otras condiciones de luz o ángulo.
