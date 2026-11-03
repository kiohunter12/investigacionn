"""
finalizar_particion_v2.py
-----------------------------------------------------------------------
Paso 2 de la reconstruccion de la particion de Olist (ver
reconstruir_particion_olist.py para el paso 1, que genera
olist_train_v2.csv / olist_val_v2.csv / olist_test_v2.csv /
olist_id_maps_v2.json a partir de la logica sin fuga verificada
en el short paper: particion por PERSONA y por MOMENTO DE COMPRA).

Este script:
  1. Respalda los archivos "canonicos" actuales (los que leen
     modelo_gnn.py, modelo_gnn_5semillas.py, modelo_transformer.py,
     modelo_baselines.py, modelo_hibrido.py, modelo_hibrido_gate.py,
     modelo_dual_agente.py) en data/processed/respaldo_particion_v1_leaky/,
     SIN borrarlos del directorio original de trabajo (precaucion).
  2. Reconstruye, a partir de los _v2 generados en el paso 1:
       - olist_edge_index.npy   -> SOLO con aristas de olist_train_v2
         (grafo bipartito no dirigido, misma convencion que
         preprocesar_dataset.py / corregir_grafo.py). Se construye
         directamente sin fuga, sin necesitar corregir_grafo.py
         despues.
       - olist_secuencias.json  -> por usuario, SOLO items de train_v2
         ordenados por timestamp (documentacion; modelo_transformer.py
         ya construye su propia secuencia desde train.csv, asi que esto
         no cambia sus resultados, pero se regenera por consistencia).
       - olist_interacciones.csv -> union train_v2+val_v2+test_v2
         (documentacion/trazabilidad, con user_idx/item_idx ya
         codificados de forma consistente).
  3. Sobrescribe los nombres canonicos:
       olist_train.csv, olist_val.csv, olist_test.csv,
       olist_id_maps.json, olist_edge_index.npy, olist_secuencias.json,
       olist_interacciones.csv
     para que TODOS los scripts de modelos corran sin ningun cambio de
     codigo, igual que hizo corregir_grafo.py con la fuga anterior.

Requisito: haber corrido antes
    python src/reconstruir_particion_olist.py

Uso:
    python src/finalizar_particion_v2.py
-----------------------------------------------------------------------
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

PROC_DIR = "data/processed"
BACKUP_DIR = f"{PROC_DIR}/respaldo_particion_v1_leaky"

CANONICOS = [
    "olist_train.csv", "olist_val.csv", "olist_test.csv",
    "olist_id_maps.json", "olist_edge_index.npy",
    "olist_edge_index_con_fuga.npy", "olist_secuencias.json",
    "olist_interacciones.csv", "olist_metadatos.csv",
]


def respaldar():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for nombre in CANONICOS:
        origen = f"{PROC_DIR}/{nombre}"
        if os.path.exists(origen):
            destino = f"{BACKUP_DIR}/{nombre}"
            if not os.path.exists(destino):
                shutil.copy(origen, destino)
                print(f"respaldado: {nombre} -> {destino}")
            else:
                print(f"ya respaldado (no se sobreescribe): {nombre}")


def construir_grafo_sin_fuga(train_df, n_users):
    src = train_df["user_idx"].to_numpy()
    dst = train_df["item_idx"].to_numpy() + n_users
    edge_index = np.vstack([np.concatenate([src, dst]),
                             np.concatenate([dst, src])])
    return edge_index


def construir_secuencias(train_df):
    df_sorted = train_df.sort_values(["user_idx", "timestamp"])
    return (df_sorted.groupby("user_idx")["item_idx"]
            .apply(list).to_dict())


def main():
    respaldar()

    train = pd.read_csv(f"{PROC_DIR}/olist_train_v2.csv", parse_dates=["timestamp"])
    val = pd.read_csv(f"{PROC_DIR}/olist_val_v2.csv", parse_dates=["timestamp"])
    test = pd.read_csv(f"{PROC_DIR}/olist_test_v2.csv", parse_dates=["timestamp"])
    idmaps = json.load(open(f"{PROC_DIR}/olist_id_maps_v2.json"))
    n_users = idmaps["n_users"]

    edge_index = construir_grafo_sin_fuga(train, n_users)
    secuencias = construir_secuencias(train)
    interacciones = pd.concat([train, val, test], ignore_index=True)

    train.to_csv(f"{PROC_DIR}/olist_train.csv", index=False)
    val.to_csv(f"{PROC_DIR}/olist_val.csv", index=False)
    test.to_csv(f"{PROC_DIR}/olist_test.csv", index=False)
    with open(f"{PROC_DIR}/olist_id_maps.json", "w") as f:
        json.dump(idmaps, f)
    np.save(f"{PROC_DIR}/olist_edge_index.npy", edge_index)
    with open(f"{PROC_DIR}/olist_secuencias.json", "w") as f:
        json.dump({str(k): v for k, v in secuencias.items()}, f)
    interacciones.to_csv(f"{PROC_DIR}/olist_interacciones.csv", index=False)

    print(f"\n[olist] particion v2 instalada como canonica:")
    print(f"  train={len(train):,} val={len(val):,} test={len(test):,}")
    print(f"  n_users={n_users:,} n_items={idmaps['n_items']:,}")
    print(f"  edge_index: {edge_index.shape[1]:,} aristas (SOLO de train, sin fuga)")
    print(f"  secuencias: {len(secuencias):,} usuarios con historial en train")
    print(f"\nRespaldo de la version anterior (particion leakeada) en: {BACKUP_DIR}/")
    print("\nListos para re-entrenar (en este orden, Docker/torch):")
    print("  python src/modelo_gnn_5semillas.py --dataset olist")
    print("  python src/modelo_transformer.py --dataset olist")
    print("  python src/modelo_hibrido.py / modelo_hibrido_gate.py --dataset olist")
    print("  python src/modelo_baselines.py --dataset olist")


if __name__ == "__main__":
    main()
