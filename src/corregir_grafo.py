"""
corregir_grafo.py
-----------------------------------------------------------------------
Corrige la fuga estructural detectada en la Seccion VI-A del short
paper: el edge_index.npy original (generado por preprocesar_dataset.py)
se construyo con TODAS las interacciones, incluida la interaccion de
test de cada usuario (protocolo leave-last-out). Esto le daba a
cualquier modelo basado en grafo (rama GNN, hibrido, y los baselines
LightGCN/TransGNN/SIGformer) acceso estructural directo al item
objetivo durante entrenamiento e inferencia, inflando sus metricas.

Este script reconstruye edge_index.npy usando UNICAMENTE las aristas de
{nombre}_train.csv (que ya genero particionar_dataset.py), con la misma
convencion de antes (grafo bipartito no dirigido, items desplazados
+n_users). No toca ningun otro archivo del pipeline ni la logica de
ningun modelo -- modelo_gnn.py, modelo_hibrido_gate.py y
modelo_baselines.py ya leen edge_index.npy tal cual, asi que basta con
corregir este unico archivo para que TODOS los modelos basados en grafo
queden corregidos de una sola vez.

El edge_index original (con fuga) se conserva como
{nombre}_edge_index_con_fuga.npy por si se necesita comparar.

Entrada:
    data/processed/{nombre}_train.csv   (ya generado por particionar_dataset.py)
    data/processed/{nombre}_id_maps.json
    data/processed/{nombre}_edge_index.npy   (el que tiene la fuga)

Salida:
    data/processed/{nombre}_edge_index.npy            (SOBREESCRITO, sin fuga)
    data/processed/{nombre}_edge_index_con_fuga.npy    (copia del original, para referencia)

Uso (desde /workspace dentro del contenedor Docker):
    python src/corregir_grafo.py --dataset olist
    python src/corregir_grafo.py --dataset amazon
    python src/corregir_grafo.py --dataset all

Despues de correr esto, hay que VOLVER A CORRER (en este orden, ya que
todos dependen del edge_index corregido):
    python src/modelo_gnn.py --dataset all --epochs 50
    python src/modelo_hibrido_gate.py --dataset all --epochs_ramas 50
    python src/modelo_baselines.py --dataset all --epochs 50
    python src/modelo_xai.py --dataset all --epochs_ramas 50
(modelo_transformer.py NO depende del grafo -- usa solo las secuencias
-- asi que sus resultados no deberian cambiar, pero no esta de mas
confirmarlo corriendolo tambien si hay tiempo.)
-----------------------------------------------------------------------
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

PROC_DIR = "data/processed"


def construir_grafo_sin_fuga(train_df, n_users):
    """Identica convencion que construir_grafo() en preprocesar_dataset.py,
    pero recibe el DataFrame de TRAIN en vez del dataset completo."""
    src = train_df["user_idx"].to_numpy()
    dst = train_df["item_idx"].to_numpy() + n_users
    edge_index = np.vstack([np.concatenate([src, dst]),
                             np.concatenate([dst, src])])
    return edge_index


def procesar(nombre, proc_dir=PROC_DIR):
    train_path = f"{proc_dir}/{nombre}_train.csv"
    idmaps_path = f"{proc_dir}/{nombre}_id_maps.json"
    edge_path = f"{proc_dir}/{nombre}_edge_index.npy"
    backup_path = f"{proc_dir}/{nombre}_edge_index_con_fuga.npy"

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"No se encontro {train_path}. Corre primero particionar_dataset.py "
            f"--dataset {nombre}.")

    idmaps = json.load(open(idmaps_path))
    n_users = idmaps["n_users"]
    train_df = pd.read_csv(train_path)

    edge_index_viejo = np.load(edge_path)
    print(f"[{nombre}] edge_index actual (con fuga): {edge_index_viejo.shape[1]:,} aristas")

    if not os.path.exists(backup_path):
        shutil.copy(edge_path, backup_path)
        print(f"[{nombre}] respaldo guardado en {backup_path}")
    else:
        print(f"[{nombre}] ya existe un respaldo en {backup_path}, no se sobreescribe")

    edge_index_nuevo = construir_grafo_sin_fuga(train_df, n_users)
    np.save(edge_path, edge_index_nuevo)

    print(f"[{nombre}] edge_index corregido (solo train): "
          f"{edge_index_nuevo.shape[1]:,} aristas "
          f"({edge_index_viejo.shape[1] - edge_index_nuevo.shape[1]:,} aristas de "
          f"val/test removidas) -> guardado en {edge_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("olist", "all"):
        procesar("olist")
    if args.dataset in ("amazon", "all"):
        procesar("amazon")
