# Pipeline de datos — PPI (GNN + Transformer + XAI)

Estructura del proyecto:

```
proyecto_pipeline/
├── Dockerfile            <- imagen del espacio de trabajo (Python + PyTorch + PyG + Transformers + Jupyter)
├── docker-compose.yml    <- levanta el contenedor con un solo comando
├── requirements.txt      <- librerias instaladas dentro del contenedor
├── data/
│   ├── raw/            <- coloca aqui los CSV originales (ver los .txt en cada subcarpeta)
│   │   ├── olist/
│   │   └── amazon/
│   └── processed/      <- se llena automaticamente al correr el notebook
├── graficos/            <- graficos generados automaticamente (PNG)
├── src/                 <- los dos scripts del pipeline (Anexo E del PPI)
│   ├── preprocesar_dataset.py
│   └── particionar_dataset.py
└── notebooks/
    └── 01_pipeline_datos.ipynb   <- abre este notebook en Jupyter y corre todo en orden
```

## Como usarlo (con Docker)

1. Descarga los datasets reales y colocalos en `data/raw/olist/` y `data/raw/amazon/`
   (revisa los archivos `.txt` de esas carpetas para ver los nombres exactos que se esperan).
2. Abre una terminal (PowerShell o CMD) **dentro de la carpeta `proyecto_pipeline/`** y corre:
   ```
   docker compose up --build
   ```
   La primera vez tarda unos minutos (instala PyTorch, PyTorch Geometric, Transformers, etc.).
3. Cuando el log muestre algo como `http://127.0.0.1:8888/lab`, abre esa direccion en tu navegador
   (o simplemente `http://localhost:8888`). Ahi tienes Jupyter Lab corriendo, con toda la carpeta
   del proyecto montada dentro del contenedor (cualquier archivo que edites/generes queda tambien
   guardado en tu carpeta real, no se pierde al apagar el contenedor).
4. Abre `notebooks/01_pipeline_datos.ipynb` y ejecuta las celdas en orden (Run All).
5. Los resultados quedan en `data/processed/` (CSV, .npy, .json) y los graficos en `graficos/`.
6. Para apagar el contenedor: `Ctrl+C` en la terminal, y opcionalmente `docker compose down`.

No necesitas tocar los scripts de `src/`; el notebook ya los importa y los corre. Esta es la base
que vas a reutilizar en las Semanas 3-4 para las ramas GNN (PyTorch Geometric) y Transformer
(libreria `transformers`) — ya quedan instaladas en la misma imagen.

## Alternativa sin Docker

Tambien puedes correr todo con tu Python local (`pip install -r requirements.txt` y
`jupyter lab`), sin el contenedor. Docker solo te da un entorno aislado y reproducible;
no es obligatorio si prefieres instalar las librerias directo en tu maquina.
