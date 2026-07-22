# Modelado dinámico y optimización paramétrica de pautas de dosificación de antibióticos

Material computacional asociado al Trabajo Fin de Máster de **Martha Ximena Dávalos Villegas**, desarrollado en el Máster Universitario en Ingeniería Matemática y Computación de la Universidad Internacional de La Rioja.

## Descripción

Este repositorio contiene la implementación de un modelo dinámico *within-host* para estudiar la evolución de bacterias sensibles y resistentes bajo diferentes pautas de dosificación de antibióticos.

El modelo incorpora:

- crecimiento logístico y competencia bacteriana;
- efecto farmacodinámico de tipo Hill;
- eliminación farmacocinética de primer orden;
- pautas periódicas de infusión;
- optimización mediante evolución diferencial;
- barridos paramétricos;
- aproximación discreta de soluciones no dominadas;
- análisis de respuesta estacionaria bajo exposición constante;
- análisis de sensibilidad local.

## Estructura del repositorio

- `simulaciones_tfm_revisadas.py`: código principal del modelo y de los experimentos computacionales.
- `requirements.txt`: dependencias necesarias para ejecutar el proyecto.
- `datos/`: resultados numéricos generados en formato CSV.
- `figuras/`: figuras producidas automáticamente por el código.

## Requisitos

- Python 3
- NumPy
- pandas
- Matplotlib
- SciPy

Las dependencias pueden instalarse mediante:

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Desde la carpeta principal del proyecto, ejecute:

```bash
python simulaciones_tfm_revisadas.py
```

El programa genera automáticamente los archivos de resultados y las figuras utilizadas en la memoria.

## Configuración principal

- Horizonte transitorio: 7 días.
- Método de integración: `LSODA`.
- Semilla de evolución diferencial: `2026`.
- Número máximo de generaciones: `35`.
- Tamaño relativo de la población del optimizador: `8`.
- Tolerancia del optimizador: `1e-5`.

## Resultados generados

La ejecución produce, entre otros, los siguientes archivos:

- resultados de los escenarios de referencia;
- trayectorias de bacterias sensibles, resistentes y concentración antibiótica;
- barrido unidimensional de la tasa de infusión;
- barrido bidimensional de tasa e intervalo;
- exploración de soluciones no dominadas;
- análisis de respuesta estacionaria;
- análisis de sensibilidad local;
- resumen de la ejecución del optimizador.

## Reproducibilidad

La semilla aleatoria y los parámetros principales de la simulación se encuentran definidos explícitamente en el código. Los archivos incluidos en las carpetas `datos/` y `figuras/` pueden regenerarse ejecutando el script principal.

## Alcance

Los parámetros, las concentraciones y las dosis utilizadas tienen carácter nominal y relativo. Los resultados corresponden a un estudio matemático y computacional exploratorio y no constituyen recomendaciones terapéuticas.

## Autora

**Martha Ximena Dávalos Villegas**
