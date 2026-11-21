# ElectroPV — diagnóstico de fallas fotovoltaicas

Aplicación educativa de Machine Learning para explorar mediciones experimentales fotovoltaicas. Incluye comparación supervisada y no supervisada, entradas eléctricas y térmicas, seis muestras físicas de panel y pruebas independientes.

## Qué incluye

- `app.py`: interfaz en español con guía, evaluación y carga de CSV independiente.
- `ml_core.py`: preparación de ventanas, entrenamiento, métricas y persistencia de artefactos.
- `data/csv_pruebas/F0L.csv` a `F7M.csv`: 16 CSV de características derivados de mediciones experimentales reales.
- `data/panel_cases/`: 18 CSV de seis paneles físicos distintos, cada uno bajo tres condiciones (limpio, sucio y sombreado). Total: 3.294 ventanas.
- `ElectroPV_CSV_reales_16_experimentos.zip`: paquete con los CSV de ejemplo.
- `INFORME_TECNICO.md` y `GUIA_SUSTENTACION.md`: metodología y explicación para exposición.
- `tools/build_features.mjs` y `tools/build_panel_cases.py`: procesos reproducibles para preparar ambas fuentes de datos.

## Abrir en Windows

1. Descomprime `ElectroPV_entrega.zip` en una carpeta.
2. Haz doble clic en `run_app.bat`.
3. Espera a que se abra el navegador en la aplicación local.

Se necesita Python 3.11 o superior. El primer inicio instala las bibliotecas; los inicios siguientes solo abren la aplicación. Si ya está publicada en Streamlit Community Cloud, se puede entrar desde el enlace público sin iniciar el proceso local.

## Probar los CSV

En **Probar 6 paneles**, selecciona el panel físico y la condición, o descarga los 18 CSV. Cada archivo contiene una combinación panel-condición y se puede probar sin sustituir otros. En **Probar mis CSV**, carga uno o varios archivos. En **Inicio** permanecen los 16 casos GPVS como conjunto adicional.

Para evaluar los modelos incluidos, abre **Evaluar modelos**, elige corriente, voltaje o ambas, y pulsa **Ejecutar evaluación completa**. El entrenamiento usa un modo (L o M) y la prueba usa el otro.

## Datos y atribución

Bakdi, A., Guichi, A., Mekhilef, S. y Bounoua, W. (2020). *GPVS-Faults: Experimental Data for fault scenarios in grid-connected PV systems under MPPT and IPPT modes*. Mendeley Data, V1. https://doi.org/10.17632/n76t439f65.1. Licencia CC BY 4.0.

*Experimental data for Mismatching Faults of Photovoltaic Modules*. Mendeley Data, DOI: https://doi.org/10.17632/xjs42j8dtf.1. Datos experimentales de seis muestras físicas de panel, medidas eléctricas y termografías; licencia CC BY 4.0. Son seis paneles, no seis plantas solares independientes.

Los CSV incluidos son características calculadas por este proyecto a partir de las mediciones originales. El código conserva los CSV originales fuera del paquete ligero de entrega.

