# Modelado dinámico y optimización paramétrica de pautas de dosificación de antibióticos

Material computacional asociado al Trabajo Fin de Máster de
Martha Ximena Dávalos Villegas, desarrollado en el Máster en
Ingeniería Matemática y Computación de la Universidad Internacional
de La Rioja.

## Descripción

El repositorio contiene la implementación de un modelo dinámico
within-host para estudiar la evolución de bacterias sensibles y
resistentes bajo diferentes pautas de dosificación de antibióticos.

El modelo incorpora:

- crecimiento logístico y competencia bacteriana;
- efecto farmacodinámico de tipo Hill;
- eliminación farmacocinética de primer orden;
- pautas periódicas de infusión;
- optimización mediante evolución diferencial;
- barridos paramétricos;
- aproximación discreta de soluciones no dominadas;
- análisis estacionario;
- análisis de sensibilidad local.

## Estructura

- `simulaciones_tfm_revisadas.py`: código principal.
- `requirements.txt`: dependencias de Python.
- `datos/`: resultados numéricos en formato CSV.
- `figuras/`: figuras generadas por el código.

## Requisitos

- Python 3
- NumPy
- pandas
- Matplotlib
- SciPy

Las dependencias pueden instalarse mediante:

```bash
python -m pip install -r requirements.txt
