# Guía rápida para la sustentación de ElectroPV

## Presentación de 40 segundos

“Mi proyecto se llama ElectroPV. Es una aplicación para estudiar fallas en un sistema fotovoltaico conectado a la red usando mediciones eléctricas reales de laboratorio. Trabaja con 16 registros que cubren el estado normal, siete escenarios de falla y dos modos de operación. La aplicación analiza corriente, voltaje o ambos, convierte las señales en características y compara modelos supervisados con K-means. Para la prueba supervisada entreno con archivos de un modo y evalúo con archivos completos del otro, para evitar que ventanas del mismo registro aparezcan a ambos lados. Los resultados son experimentales y no prueban por sí solos que el sistema funcione en cualquier instalación.”

## Demostración en vivo

1. Abrir **Inicio** y mostrar que hay 16 experimentos y tres modalidades seleccionables.
2. Entrar en **Evaluar modelos**; dejar todos los F0–F7 seleccionados.
3. Elegir entrenamiento L y señales Corriente + voltaje; ejecutar evaluación.
4. Mostrar la tabla de modelos y explicar la matriz de confusión: diagonal = aciertos, fuera de diagonal = confusiones.
5. Mostrar K-means; decir que agrupa sin etiquetas y que ARI/silhouette tienen significados distintos a exactitud.
6. Abrir **Probar mis CSV**, descargar desde Inicio el paquete y cargar un CSV F2M para mostrar cómo se procesa de forma independiente.
7. Terminar en **Sustentación** o **Entender el proyecto** para responder preguntas.

## Respuestas cortas

**¿Son reales los datos?** Sí, son mediciones de experimentos de laboratorio publicadas por los autores en Mendeley Data. No son de una planta comercial.

**¿Qué es multimodal?** Uso dos tipos de señal, corriente y voltaje. Puedo probarlos por separado y juntos.

**¿Qué hace Random Forest?** Combina las decisiones de varios árboles para clasificar cada fragmento de señal.

**¿Qué hace K-means?** Agrupa mediciones parecidas sin conocer de antemano sus etiquetas. Luego reviso si los grupos coinciden con las fallas conocidas.

**¿Qué es una ventana?** Un bloque de 1.024 mediciones seguidas. Calculo cinco resúmenes para cada una de las 13 señales.

**¿Qué significa F0L / F2M?** F0 es normal; F2 es falla del inversor. L es potencia limitada (IPPT) y M máxima potencia (MPPT).

**¿Por qué no mezclar ventanas aleatoriamente?** Porque ventanas del mismo experimento podrían quedar en entrenamiento y prueba y hacer parecer que el modelo funciona mejor de lo que realmente generaliza. Se reservan archivos completos de otro modo.

**¿Se aplica a cualquier sistema solar?** Todavía no se puede asegurar. Se necesitarían datos externos de varias instalaciones para probarlo.

**¿Qué es silhouette?** Una medida de qué tan separados/compactos quedan los grupos. No es exactitud.

**¿Qué es ARI?** Una comparación posterior entre los grupos de K-means y las etiquetas reales. Las etiquetas no se usaron para crear los grupos.

## Referencia bibliográfica

Bakdi, A., Guichi, A., Mekhilef, S. y Bounoua, W. (2020). *GPVS-Faults: Experimental Data for fault scenarios in grid-connected PV systems under MPPT and IPPT modes*. Mendeley Data, V1. DOI: 10.17632/n76t439f65.1. Licencia CC BY 4.0.

