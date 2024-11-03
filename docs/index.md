# Ejercicio 2 de la prueba de Data Science

¡Hola! Bienvenido a mi propuesta de solución para el primer ejercicio de la prueba. A continuación encontrarás la documentación necesaria para poder entender el planteamiento del problema y la construcción del paquete para la puesta en producción del modelo.

## El Dataset Escogido

El dataset escogido es [Electricity Transformer Dataset](https://github.com/zhouhaoyi/ETDataset), donde tendremos que predecir la temperatura del aceite dentro de un transformador eléctrico a partir de una serie de indicadores de carga, con datos correspondientes a dos años recogidos cada hora. En este caso utilizaremos la versión del dataset accesible a través de la libería *darts* como [ETTh1Dataset](https://unit8co.github.io/darts/generated_api/darts.datasets.html#darts.datasets.ETTm1Dataset).

*¿Por qué este Dataset?*

He escogido este Dataset porque presenta un problema de regresión de una serie temporal multivariante, pero con un tamaño asequible para este ejercicio. 

## Planteamiento del ejercicio

Se han seguido los siguientes pasos a la hora de abordar el problema:

1. En primer lugar se ha llevado acabo un análisis exploratorio de la variable objetivo y las covariables, incluyendo el análisis de la estacionalidad de la serie a predecir. A partir de este análisis determinaremos las posibles limitaciones o condiciones a tener en cuenta al seleccionar un modelo, así como las  transformaciones necesarias. Este paso se puede ver en el archivo `notebooks/analisis-exploratorio.ipynb`, o también embebido en esta documentación en la sección [Analisis exploratorio](analisis-exploratorio.ipynb).
2. A continuacion se realiza la selección del modelo en varias fases. En primer lugar se realizan las transformaciones necesarias de los datos. A continuación, se plantean los modelos a considerar, la métrica de evaluación y se realiza el tuning de hiperparámetros mediante un proceso de backtesting. A partir de los modelos ya tuneados, predecimos los últimos periodos de la serie que hemos reservado y a partir de la puntuación de esas predicciones seleccionamos el modelo a considerar. Una vez decidido el modelo fina, lo entrenamos y predecimos 100 periodos en adelante, sin conocer el valor de las covariables. Este paso se puede ver en el archivo `notebooks/transformacion-seleccion-modelo.ipynb`, o también embebido en esta documentación en la sección [Seleccion del modelo](seleccion-modelo.ipynb).
3. Por último, creamos un paquete de Python que contenga el modelo final, la lógica del entrenamiento y la predicción de nuevos datos. Para más detalles sobre la organización del código, consultar el fichero README.md de este repositorio y para la guía de instalación de dependencias y uso del código, ver la sección [Instalación de dependencias y Uso](instalacion-y-uso.md) de esta documentación.
