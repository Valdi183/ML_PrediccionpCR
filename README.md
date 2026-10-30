# Predicción de pCR en BreastDCEDL

Red neuronal convolucional propia, entrenada desde cero, que a partir de una resonancia de
mama con contraste (DCE-MRI) tomada **antes** de la quimioterapia estima la probabilidad de
**pCR**: que tras el tratamiento no quede tumor invasivo. No es detección de tumores: es
pronóstico de la respuesta a un tratamiento que todavía no ha empezado.

**Web (la app):** <https://valdi183.github.io/ML_PrediccionpCR/>: el caso explicado
(`index.html`) y la predicción en el navegador (`predecir.html`, ONNX Runtime Web; las
imágenes no salen del ordenador). Uso educativo: **sin validez clínica**.

| | train | test | total |
|---|---|---|---|
| pacientes | 1.097 | 176 | **1.273** |
| cortes | 10.945 | 1.758 | **12.703** |
| imágenes PNG | 32.835 | 5.274 | **38.109** |

Un corte son tres PNG de 256×256 en grises: `_PRE`, `_EARLY` y `_LATE`, tres instantes del
contraste (no RGB; `z012` es una altura, no un tiempo). El 70,6 % de las pacientes son
pCR=0. Tres cohortes (I-SPY1, I-SPY2 y Duke) procesadas de forma distinta.

**Resultado.** Conjunto de 5 CNN (una por fold), 18.329 parámetros cada una. Validación
(5 folds): AUC medio de todas las épocas 0,551 ± 0,011. **Test, evaluado una sola vez:** por
paciente AUC 0,544 [0,457-0,629], sensibilidad 0,038, especificidad 0,935, F1 macro 0,430
con el umbral 0,3635 fijado en validación. La meta realista de la asignatura es 0,60-0,70
(referencia de la asignatura: AUC ≈ 0,64 en test); Fridman et al. 2025 (0,72 en el mismo
test) usa un modelo preentrenado, así que es una cota superior y no es comparable. Todo lo
decidido y por qué: [`docs/decisiones.md`](docs/decisiones.md); todo lo probado:
[`docs/iteraciones.md`](docs/iteraciones.md).

El historial de desarrollo (todos los commits, la primera versión de la app y los
lanzadores del laboratorio) está en el repositorio anterior,
[`Valdi183/ML_Caso_1_Cancer`](https://github.com/Valdi183/ML_Caso_1_Cancer). Este es el
repositorio de entrega.

---

## Reglas del caso

Invalidan el trabajo, y se han respetado:

1. **CNN propia desde cero**, una sola salida: nada preentrenado ni transfer learning.
2. **Separación por paciente** con la columna `fold`: ningún corte de una paciente a la vez
   en entrenamiento y validación.
3. **`test` una sola vez, al final** (`results/test/`); nada se ajustó con él.
4. **No publicar la validación privada** ni sus etiquetas: no está en el repositorio.

---

## Puesta en marcha

Windows (PowerShell), desde la raíz:

```powershell
git clone https://github.com/Valdi183/ML_PrediccionpCR.git
cd ML_PrediccionpCR
.\configurar.ps1            # entorno, PyTorch según haya GPU NVIDIA o no, dataset y verificación
```

`configurar.ps1` es idempotente (si se corta, se vuelve a lanzar). `-SinDatos` prepara solo el
entorno; `-Datos D:\bdcedl` elige la carpeta del dataset. A mano, en cualquier sistema:

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # o cu124 con GPU NVIDIA
pip install -r requirements.txt -r requirements-web.txt ruff pytest
python -m src.utils.verificar_entorno
```

Todo se ejecuta **desde la raíz** como módulo (`python -m src.<...>`); las rutas salen de
[`src/config.py`](src/config.py).

### El dataset y `utils_caso.py`

El dataset **no está en el repositorio** (1,27 GB, 38.109 ficheros) y no debe estarlo. Se
descarga del bucket público de la asignatura:

```bash
python -m src.data.descargar --destino C:\bdcedl --hilos 32
```

La descarga incluye `utils_caso.py`, el módulo que da la asignatura con el dataset
(particiones por paciente, carga de imágenes, agregación de cortes a paciente). **Vive en la
carpeta del dataset, no en el repositorio**, y no se copia aquí. `config.py` encuentra la
carpeta en este orden: la variable de entorno `BDCEDL`; el fichero `.bdcedl_ruta` en la raíz
(lo escribe `configurar.ps1`; no se versiona); rutas habituales (`data/bdcedl`,
`../breastdcedl`, `C:/bdcedl`, `~/bdcedl`, `D:/bdcedl`). La da por buena si contiene
`metadata/samples.csv` y `utils_caso.py`, y `config.utils()` lo importa:

```python
from src import config
uc = config.utils()
entrenamiento, validacion = uc.particion(uc.cargar_samples(), fold_val=0)
```

Sin el dataset funcionan la web, `python -m src.inferencia`, las pruebas (las que lo
necesitan se saltan) y todo lo que solo lee `results/`.

---

## Reproducir

### 1. Entrenar los 5 modelos

Configuración congelada (de `experimentos/it6.txt` y `experimentos/it8.txt`): 3 bloques
8-16-32, resumen por media, imagen a 128 px dentro de la red, pérdida normal, semilla 42 y el
resto de `config.DEFECTO`.

```bash
O="--aumentado --dropout 0.7 --wd 1e-3 --reducir 2 --filtros 8,16,32"
python -m src.training.entrenar $O --fold 0 --nombre it6_3bloques_f0
python -m src.training.entrenar $O --fold 1 --nombre it6_3bloques_f1
python -m src.training.entrenar $O --fold 2 --nombre it8_3bloques_f2
python -m src.training.entrenar $O --fold 3 --nombre it8_3bloques_f3
python -m src.training.entrenar $O --fold 4 --nombre it8_3bloques_f4
```

Cada ejecución deja `models/<nombre>.pt` y `results/<nombre>/` (`config.json`,
`historial.csv`, `metricas.json`, `predicciones_val.csv`, `curvas.png`).
`python -m src.training.entrenar --help` lista las opciones; `--rapido` es una prueba de
minutos. `experimentos/it*.txt` son las listas de cada iteración: cada línea son los
argumentos de una llamada a `src.training.entrenar`. La comparación con pérdida ponderada
son las mismas órdenes con `--ponderada --nombre it8_3bloques_pond_f<fold>`.

Los 5 pesos entregados están en `models/` y su SHA-256 en `results/test/plan.json`. En GPU,
cuDNN no es determinista: reentrenar da pesos parecidos, no idénticos.

Resumen fuera de fold (sin tocar test):

```bash
python -m src.utils.analisis_cv --config 3bloques it6_3bloques_f0 it6_3bloques_f1 \
    it8_3bloques_f2 it8_3bloques_f3 it8_3bloques_f4
python -m src.utils.comparar          # candidatos de la iteración 9 (results/comparacion/)
```

### 2. Umbral

```bash
python -m src.utils.umbral            # results/umbral/resumen.json, campo umbral_final = 0,3635
```

Criterio clínico: especificidad ≥ 0,90 por paciente (media de sus cortes). Regla: percentil
90 de la probabilidad de todas las pacientes pCR=0 fuera de fold juntas. Detalle en
`docs/decisiones.md`.

### 3. Test (hecho una vez)

```bash
python -m src.utils.evaluar_test      # se niega a repetirse: results/test/resumen.json ya existe
```

Evalúa el conjunto de los 5 modelos (media de probabilidades) con el plan fijado antes en
`results/test/plan.json` (pesos y su SHA-256, umbral, agregación). Después:
`python -m src.utils.metricas_extra` (métricas descriptivas desde las predicciones
guardadas, sin volver a pasar test por la red) y `python -m src.utils.combinado` (clínica +
CNN, solo validación, como comparación).

### 4. Figuras

`notebooks/comparativa_modelos.ipynb` reúne las curvas de todas las iteraciones y las
figuras del modelo final (cálculos en `src/utils/figuras.py`; escribe
`docs/figuras/comparativa_*.png`). Se abre desde la raíz con Jupyter (`pip install jupyter`).
La celda de los mapas de atención necesita el dataset y dos pesos de la iteración 7 que no
se entregan; ya está ejecutada y su figura guardada. `notebooks/eda.ipynb` es el análisis
exploratorio.

### 5. Web

```bash
python -m src.web.exportar              # web/modelo/conjunto.onnx, web/modelo.json y web/datos.json
python -m src.web.exportar --solo-datos # solo las cifras (datos.json, historial.json, umbral_validacion.json)
cd web && python -m http.server         # http://localhost:8000/
```

- **Ninguna cifra está escrita en el HTML ni en el JS**: salen de `web/datos.json` y
  `web/modelo.json`, generados desde `results/`, la red y `docs/data/referencias.json`.
- **Autosuficiente:** ONNX Runtime Web, `pako`, `UPNG` (`web/js/vendor/`, con sus licencias)
  y la tipografía Atkinson Hyperlegible (SIL OFL, `web/fonts/`) van en el repositorio; sin
  CDN y con rutas relativas.
- **Mismo preprocesado que el entrenamiento:** el preprocesado va dentro de la red
  (`src/models/cnn.py`), así que la entrada es el PNG tal cual / 255. `web/js/png.js` lee y
  valida el PNG con las mismas reglas y mensajes que `src/inferencia.py`: rechaza lo que no
  sea PNG, otro tamaño, más de 2 MB, ficheros corruptos, color y 16 bits.
- **Comprobar con el pipeline** (en la defensa): la misma probabilidad que muestra la web.

  ```bash
  python -m src.inferencia --pre X_PRE.png --early X_EARLY.png --late X_LATE.png
  ```

- **Despliegue:** `.github/workflows/pages.yml` publica `web/` en GitHub Pages en cada push a
  `main` que toque `web/` (en GitHub: *Settings → Pages → Source: GitHub Actions*).
- **Avisos:** el umbral se eligió por paciente (media de ~10 cortes) y la web recibe un solo
  corte. Los 4 ejemplos de `web/ejemplos/` son pacientes de train (4 de los 5 modelos las
  vieron): sirven para probar la web, no para medirla.

---

## Pruebas y CI

```bash
ruff check .
pytest -q
```

Sin el dataset, las pruebas que lo necesitan se saltan. `tests/test_web.py` compara el ONNX
con PyTorch (< 1e-5 y misma clase) y comprueba que `web/datos.json` está al día. Con
navegadores reales (`pip install playwright && python -m playwright install firefox`):
`python -m src.web.probar_navegador` (lectura del PNG igual que Python en 76 casos, mapa y
curva en 24 cortes y los 4 ejemplos a 4 decimales). `.github/workflows/ci.yml` ejecuta ruff
y pytest en cada push y pull request a `main`.

---

## Estructura

```
ML_PrediccionpCR/
├── .github/workflows/     ← ci.yml (ruff + pytest) y pages.yml (publica web/)
├── configurar.ps1         ← prepara una máquina Windows de cero
├── data/                  ← sitio opcional para el dataset (no se versiona)
├── docs/                  ← decisiones.md, iteraciones.md, figuras/, data/referencias.json
├── experimentos/          ← argumentos de cada entrenamiento, por iteración
├── models/                ← los 5 pesos del modelo final (.pt)
├── notebooks/             ← eda.ipynb, comparativa_modelos.ipynb, localizacion_tumor.ipynb
├── presentations/         ← diapositivas
├── results/               ← una carpeta por entrenamiento; umbral/, test/, comparacion/...
├── src/
│   ├── config.py          ← rutas, semilla e hiperparámetros, localización del dataset
│   ├── data/              ← descarga, lectura del PNG, carga en memoria, caja del tumor
│   ├── models/cnn.py      ← la CNN y su preprocesado (única definición)
│   ├── training/          ← entrenar.py
│   ├── inferencia.py      ← predicción del conjunto y validación de la entrada
│   ├── visualizacion.py   ← mapa y curva de captación (descriptivos)
│   ├── utils/             ← umbral, test, comparación, métricas, figuras, verificación
│   └── web/               ← exportar a ONNX y datos.json, casos de prueba del navegador
├── tests/
├── web/                   ← la web estática (GitHub Pages)
└── requirements.txt, requirements-web.txt, ruff.toml, pytest.ini
```

---

## Licencia, atribución y aviso

Las imágenes (los 4 ejemplos de `web/ejemplos/`) son imágenes desidentificadas derivadas de
**BreastDCEDL** (Fridman et al., 2025, arXiv:2506.12190), construido a partir de las
colecciones públicas I-SPY1, I-SPY2 y Duke Breast Cancer MRI, con licencia
**[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**: uso docente y de
investigación, sin fines comerciales, conservando esta atribución. Las bibliotecas de
`web/js/vendor/` y la tipografía de `web/fonts/` llevan su propia licencia junto a ellas.

**Esto no es un dispositivo médico.** No se permite reidentificación, diagnóstico ni
recomendación terapéutica. Un buen resultado retrospectivo no demuestra seguridad,
causalidad ni utilidad clínica prospectiva.
