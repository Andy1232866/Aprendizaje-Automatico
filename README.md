# Aprendizaje-Automatico

Este repositorio fue creado con el objetivo de almacenar los código desarrollados para la materia de "Aprendizaje Automático"

## Sobre este Repositorio

En éste respositorio se encuentran códigos realizados durante la materia, poniento en práctica los temas teóricos y diversos.

Podrás encontrar la siguiente estructura y archivos:

- SVM:
  - breast_cancer.csv: Archivo CSV utilizado para entrenar el modelo de máquinas de vectores de soporte
  - svm.py: Script de python el cuál se define el modelo, entrena y predice según las características dadas
  - Objetivo: Predecir si un tumor es maligno o benigno dadas características distintas (nuevas) a las ya sabidas (entrenadas)

- Redes Neuronales:
  - FashionMNIST.py: Script de python en el cual se define un modelo de redes neuronales, se entrena y predice la categoría de ropa según las imágenes
  - RN.py: Script de python en el cual es capaz de aprender el comportamiento de una compuerta lógica AND y OR según las entradas definidas y las salidas
  - Objetivo: Lograr que la computadora pueda aprender patrones en datos e imágenes con datos controlados

- K-Means:
  - analisis.csv: Archivo CSV utilizado para entrenar el modelo de K Medias
  - k-means.py: Script de python el cuál se encarga del agrupamiento de los datos según los aprendidos del CSV
  - reporte_por_cluster.xlsx: Archivo de hoja de cálculo el cuál fue utilizado para identificar las agrupaciones que éste algoritmo fue campaz de generar
  - K-Means.pdf: Arhcivo en formato PDF de explicación de éste algoritmo
  - Objetivo: Agrupar a un conjunto de personas que al humano pueda ser aleatorio, pero para la computadora tiene sentido (arupamiento por características)

- CNN:
  - preprocesamiento.py: Script de python el cual se encarga de preprocesar las imágenes con las cuales el algoritmo será entrenado (las imagénes son resonancias magnéticas cerebrales sobre 3 tipos de tumores)
  - redN.py: Script de python el cual se encarga de predecir el tipo de tumor a partir de una imagen cerebral (visión por computadora)
  - Objetivo: Lograr que la computadora pueda aprender a "ver" imagenes y clasficiarlas con datos no controlados (imagenes rotadas, recortadas, no completas, borrosas, etc)

- AD-RF:
  - Depresion.csv: Archivo CSV utilizado para entrenar los modelos de Árboles de Decisión y Random Forest
  - arbol_decision.py: Script de python en el que se emplean los árboles de decisión para determinar si una persona padece depresión o no, según sus características
  - arbol_decision_final.png: Imagen del resultado del árbol de decisión con todas sus "ramas" posibles para determinar un caso o el otro
  - random_forest.py: Script de python en el que se emplean los bosques aleatorios para determinar si una persona padece depresión o no, según sus características
  - Objetivo: Entender las características, pros, contras y en qué momento usar un algoritmo o el otro
