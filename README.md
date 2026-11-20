# DuoSense · versión para alojamiento web

Este paquete contiene solo lo necesario para ejecutar DuoSense en un servicio Streamlit:

- `app.py` y `ml_core.py`: interfaz y lógica de análisis.
- `requirements.txt`: componentes de Python necesarios.
- `data/paderborn_real_features.csv` y `data/paderborn_real_examples.npz`: datos derivados incluidos con la aplicación.

No incluye la presentación ni documentos con datos personales del estudiante.

## Atribución y uso de los datos

Los datos experimentales de Paderborn se distribuyen bajo CC BY-NC 4.0 para uso académico no comercial. Referencia: Christian Lessmeier et al., *KAt-DataCenter: Condition Monitoring of Bearing Damage in Electromechanical Drive Systems Using Motor Current Signals: A Benchmark Data Set for Data-Driven Classification*, Paderborn University (2016). Bearing DataCenter: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter.

## Alcance

La aplicación es un prototipo académico. El conjunto incluido contiene dos rodamientos físicos (uno saludable y uno con daño); las métricas no demuestran que el modelo generalice a otros rodamientos.

