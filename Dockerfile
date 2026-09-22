# Espacio de trabajo del PPI (GNN + Transformer + XAI)
# Imagen base ligera con Python 3.11
FROM python:3.11-slim

# Dependencias de sistema minimas (compilar algunas ruedas de torch-geometric)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

# Arranca Jupyter Lab sin token/contraseña (uso local, solo tu maquina)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
