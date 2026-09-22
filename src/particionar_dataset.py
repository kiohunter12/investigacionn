"""
particionar_dataset.py
-----------------------------------------------------------------------
Particion leave-last-out (Anexo E del PPI) sobre los datasets ya
procesados por preprocesar_dataset.py.

Para cada usuario, ordenado cronologicamente por timestamp:
    - la ULTIMA interaccion  -> test
    - la PENULTIMA interaccion -> validacion
    - el resto (>=1 interaccion) -> entrenamiento

Solo se particionan usuarios con al menos 3 interacciones (para poder
separar train/val/test); los usuarios con menos interacciones que el
minimo se reportan como "cold-start" y se excluyen de val/test (quedan
solo en train si tienen >=1 interaccion, para no perder la senal).

Entrada (generada por preprocesar_dataset.py):
    data/processed/{nombre}_interacciones.csv
        columnas: user_id, item_id, rating, timestamp, user_idx, item_idx

Salida:
    data/processed/{nombre}_train.csv
    data/processed/{nombre}_val.csv
    data/processed/{nombre}_test.csv
    data/processed/metadatos.csv   (una fila por dataset, se va acumulando)

Uso:
    python particionar_dataset.py --dataset olist
    python particionar_dataset.py --dataset amazon
    python particionar_dataset.py --dataset all
-----------------------------------------------------------------------
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 42
MIN_PARA_PARTICIONAR = 3  # usuario necesita >=3 interacciones para train/val/test

PROC_DIR = "data/processed"


def set_seed(seed=SEED):
    np.random.seed(seed)


def cargar_interacciones(nombre, proc_dir=PROC_DIR):
    path = os.path.join(proc_dir, f"{nombre}_interacciones.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontro {path}. Corre primero preprocesar_dataset.py "
            f"--dataset {nombre}."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def particionar_leave_last_out(df, min_para_particionar=MIN_PARA_PARTICIONAR):
    """Devuelve (train_df, val_df, test_df, n_cold_start)."""
    df_sorted = df.sort_values(["user_idx", "timestamp"])
    train_rows, val_rows, test_rows = [], [], []
    n_cold_start = 0

    for _, grupo in df_sorted.groupby("user_idx", sort=False):
        n = len(grupo)
        if n < min_para_particionar:
            # cold-start relativo: todo va a entrenamiento, nada a val/test
            train_rows.append(grupo)
            n_cold_start += 1
            continue
        train_rows.append(grupo.iloc[:-2])
        val_rows.append(grupo.iloc[[-2]])
        test_rows.append(grupo.iloc[[-1]])

    train_df = pd.concat(train_rows).reset_index(drop=True) if train_rows else \
        df_sorted.iloc[0:0]
    val_df = pd.concat(val_rows).reset_index(drop=True) if val_rows else \
        df_sorted.iloc[0:0]
    test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else \
        df_sorted.iloc[0:0]
    return train_df, val_df, test_df, n_cold_start


def guardar_metadatos(nombre, df, train_df, val_df, test_df, n_cold_start,
                       proc_dir=PROC_DIR):
    fila = {
        "dataset": nombre,
        "interacciones_totales": len(df),
        "usuarios": df["user_idx"].nunique(),
        "items": df["item_idx"].nunique(),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "usuarios_cold_start_excluidos_de_val_test": n_cold_start,
        "seed": SEED,
    }
    meta_path = os.path.join(proc_dir, "metadatos.csv")
    fila_df = pd.DataFrame([fila])
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        meta = meta[meta["dataset"] != nombre]  # reemplaza fila anterior del mismo dataset
        meta = pd.concat([meta, fila_df], ignore_index=True)
    else:
        meta = fila_df
    meta.to_csv(meta_path, index=False)
    return fila


def procesar(nombre, proc_dir=PROC_DIR):
    set_seed(SEED)
    print(f"[{nombre}] cargando interacciones procesadas...")
    df = cargar_interacciones(nombre, proc_dir)

    train_df, val_df, test_df, n_cold_start = particionar_leave_last_out(df)

    train_df.to_csv(f"{proc_dir}/{nombre}_train.csv", index=False)
    val_df.to_csv(f"{proc_dir}/{nombre}_val.csv", index=False)
    test_df.to_csv(f"{proc_dir}/{nombre}_test.csv", index=False)
    fila = guardar_metadatos(nombre, df, train_df, val_df, test_df,
                              n_cold_start, proc_dir)

    print(f"[{nombre}] train={len(train_df):,} | val={len(val_df):,} | "
          f"test={len(test_df):,} | usuarios cold-start "
          f"(<{MIN_PARA_PARTICIONAR} interacciones, excluidos de val/test)="
          f"{n_cold_start:,}")
    print(f"[{nombre}] metadatos.csv actualizado -> {fila}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"],
                         default="all")
    args = parser.parse_args()

    if args.dataset in ("olist", "all"):
        procesar("olist")
    if args.dataset in ("amazon", "all"):
        procesar("amazon")
