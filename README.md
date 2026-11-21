# ElectroPV — diagnóstico de fallas fotovoltaicas

Aplicación académica para explorar y comparar modelos de Machine Learning con señales eléctricas experimentales de un sistema fotovoltaico conectado a la red.

## Incluye
- Clasificación supervisada: árbol de decisión, Random Forest y regresión logística.
- Agrupamiento no supervisado: K-means, silhouette y ARI.
- Tres modalidades: corriente, voltaje y combinación.
- Evaluación por archivos completos, entrenando en un modo y probando en el otro.
- Dieciséis CSV compactos listos para cargar y probar.
- Carga de archivos nuevos, visualización y guardado de métricas/modelos.

## Ejecutar
Instala los paquetes de `requirements.txt` y ejecuta `streamlit run app.py`.
En **Evaluar modelos** compara algoritmos; en **Probar mis CSV** carga uno o varios casos de forma independiente.

## Datos y atribución
Bakdi, A., Guichi, A., Mekhilef, S. y Bounoua, W. (2020). *GPVS-Faults: Experimental Data for fault scenarios in grid-connected PV systems under MPPT and IPPT modes*. Mendeley Data, V1. DOI: 10.17632/n76t439f65.1. Licencia CC BY 4.0.
Fuente: https://data.mendeley.com/datasets/n76t439f65/1

Los datos proceden de ensayos de laboratorio. Los resultados no prueban generalización a cualquier planta fotovoltaica.
