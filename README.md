# Paquete de revisión del Capítulo 4

## Contenido
- `04_desarrollo_contribucion_corregido.tex`: capítulo revisado.
- `simulaciones_tfm_revisadas.py`: código reproducible.
- `figuras/`: nueve figuras generadas automáticamente.
- `datos/`: CSV de escenarios, barridos, Pareto, sensibilidad y umbrales.
- `requirements.txt`: dependencias.

## Ejecución
```bash
python -m pip install -r requirements.txt
python simulaciones_tfm_revisadas.py
```

## Pauta optimizada reproducida
- q = 6.22677670
- tau = 1.74999886
- delta = 0.41297911
- J = 0.01516948
- evaluaciones = 1084

El optimizador alcanzó el número máximo de iteraciones; por eso se reporta como la mejor solución encontrada, no como una demostración del mínimo global exacto.

## Hallazgos críticos
1. La dosis se calcula exactamente a partir de las ventanas de infusión.
2. La normalización original hace que el término resistente de J sea numéricamente pequeño.
3. El umbral resistente bajo entrada constante es q_R = 29.2404, fuera del dominio q <= 12.
4. El análisis de Pareto usa una malla sistemática de 918 pautas.
5. Debe añadirse el enlace real al repositorio antes de entregar la memoria.
