"""
preprocesar_dataset.py
-----------------------------------------------------------------------
Pipeline de preprocesamiento para el proyecto PPI:
"Modelo hibrido de Graph Neural Networks y Transformer con
explicabilidad (XAI) para sistemas de recomendacion multiagente en
e-commerce latinoamericano" (Anexo E del PPI).

Para Olist (Brasil, dataset principal) y Amazon Beauty (control
internacional) construye:
  1. La senal de interaccion usuario-producto (con timestamp).
  2. El filtro de usuarios/items con pocas interacciones (umbral
     diferenciado por dataset: Olist >=2, Amazon Beauty >=5 -- ver
     MIN_INTERACTIONS_OLIST / MIN_INTERACTIONS_AMAZON mas abajo. Olist
     es mucho mas disperso a nivel de SKU: ~97% de sus clientes compran
     un unico producto en todo el dataset, por lo que un umbral de 5
     para ambos deja Olist vacio; se documenta como limitacion).
  3. La codificacion de IDs a indices enteros consecutivos.
  4. El grafo usuario-producto (edge_index estilo PyTorch Geometric).
  5. Las secuencias temporales por usuario (entrada del Transformer).

Estructura de carpetas esperada:
    data/raw/olist/olist_orders_dataset.csv
    data/raw/olist/olist_order_items_dataset.csv
    data/raw/olist/olist_customers_dataset.csv       (necesario: customer_unique_id)
    data/raw/olist/olist_order_reviews_dataset.csv   (opcional)
    data/raw/amazon/ratings_Beauty.csv

Uso:
    python preprocesar_dataset.py --dataset olist
    python preprocesar_dataset.py --dataset amazon
    python preprocesar_dataset.py --dataset all
-----------------------------------------------------------------------
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SEED = 42
# Umbral minimo de interacciones (Seccion 2.5 / Anexo E). Olist es mucho
# mas disperso que Amazon Beauty a nivel de producto individual (SKU):
# ~97% de los clientes de Olist compran un unico producto en todo el
# dataset, asi que el umbral de 5 (pensado originalmente para ambos
# datasets por igual) deja Olist vacio. Se documenta como limitacion:
# se usa un umbral mas bajo para Olist y se mantiene 5 para Amazon Beauty
# (que si tiene suficiente densidad de repeticion).
MIN_INTERACTIONS_OLIST = 2
MIN_INTERACTIONS_AMAZON = 5

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"


def set_seed(seed=SEED):
    np.random.seed(seed)


# ------------------------------------------------------------------
# 1. Carga y construccion de la interaccion usuario-producto
# ------------------------------------------------------------------

def cargar_interacciones_olist(raw_dir=RAW_DIR):
    """Cruza orders + order_items (+ reviews) para construir la senal
    de interaccion usuario-producto de Olist (compra confirmada),
    enriquecida con la calificacion cuando exista (Seccion 2.5).

    IMPORTANTE: en Olist, `customer_id` (tabla orders) es un identificador
    *por pedido*, casi 1 a 1 con order_id (no identifica al cliente real
    entre compras distintas). El cliente recurrente real es
    `customer_unique_id`, en olist_customers_dataset.csv. Por eso este
    loader cruza tambien con esa tabla y usa customer_unique_id como
    user_id; si se usara customer_id, el filtro de <5 interacciones
    (Seccion 2, todo usuario aparece con 1 sola interaccion) deja el
    dataset vacio."""
    path = os.path.join(raw_dir, "olist")
    orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
    customers = pd.read_csv(os.path.join(path, "olist_customers_dataset.csv"))
    reviews_path = os.path.join(path, "olist_order_reviews_dataset.csv")

    orders = orders[["order_id", "customer_id", "order_status",
                      "order_purchase_timestamp"]]
    items = items[["order_id", "product_id"]]
    customers = customers[["customer_id", "customer_unique_id"]]

    orders = orders.merge(customers, on="customer_id", how="left")
    inter = items.merge(orders, on="order_id", how="inner")
    # compra confirmada: se excluyen pedidos cancelados / no disponibles
    inter = inter[~inter["order_status"].isin(["canceled", "unavailable"])]

    if os.path.exists(reviews_path):
        reviews = pd.read_csv(reviews_path)[["order_id", "review_score"]]
        reviews = reviews.drop_duplicates(subset="order_id")
        inter = inter.merge(reviews, on="order_id", how="left")
    else:
        inter["review_score"] = np.nan

    inter["timestamp"] = pd.to_datetime(inter["order_purchase_timestamp"])
    inter = inter.rename(columns={"customer_unique_id": "user_id",
                                   "product_id": "item_id",
                                   "review_score": "rating"})
    inter = inter[["user_id", "item_id", "rating", "timestamp"]].dropna(
        subset=["user_id", "item_id", "timestamp"])
    inter = inter.drop_duplicates(subset=["user_id", "item_id", "timestamp"])
    return inter.reset_index(drop=True)


def cargar_interacciones_amazon(raw_dir=RAW_DIR):
    """Lee ratings_Beauty.csv (formato SNAP/McAuley: sin encabezado,
    columnas user,item,rating,timestamp en epoch Unix)."""
    path = os.path.join(raw_dir, "amazon", "ratings_Beauty.csv")
    df = pd.read_csv(path, header=None,
                      names=["user_id", "item_id", "rating", "timestamp"])
    # algunos mirrors sí traen encabezado de texto en la primera fila
    if not pd.api.types.is_numeric_dtype(
            pd.to_numeric(df["rating"], errors="coerce")):
        df = df.iloc[1:].reset_index(drop=True)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.drop_duplicates(subset=["user_id", "item_id", "timestamp"])
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# 2. Filtro de cold-start extremo (<5 interacciones, Seccion 2.5)
# ------------------------------------------------------------------

def filtrar_min_interacciones(df, min_int):
    """Filtra iterativamente usuarios e items con menos de `min_int`
    interacciones, hasta que el conjunto se estabiliza (quitar un item
    puede dejar a un usuario por debajo del umbral, y viceversa)."""
    prev_len = -1
    while prev_len != len(df):
        prev_len = len(df)
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_int].index)]
        df = df[df["item_id"].isin(item_counts[item_counts >= min_int].index)]
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# 3. Codificacion de IDs a indices enteros consecutivos
# ------------------------------------------------------------------

def codificar_ids(df):
    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {p: i for i, p in enumerate(item_ids)}
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["item_idx"] = df["item_id"].map(item2idx)
    return df, user2idx, item2idx


# ------------------------------------------------------------------
# 4. Grafo usuario-producto (edge_index estilo PyTorch Geometric)
# ------------------------------------------------------------------

def construir_grafo(df, n_users):
    """edge_index de forma (2, E): nodos usuario indexados 0..n_users-1
    y nodos item desplazados +n_users, para compartir un unico espacio
    de nodos en un grafo bipartito usuario-item (convencion habitual
    en PyTorch Geometric). El grafo se construye no dirigido."""
    src = df["user_idx"].to_numpy()
    dst = df["item_idx"].to_numpy() + n_users
    edge_index = np.vstack([np.concatenate([src, dst]),
                             np.concatenate([dst, src])])
    return edge_index


# ------------------------------------------------------------------
# 5. Secuencias temporales por usuario (input del Transformer)
# ------------------------------------------------------------------

def construir_secuencias(df):
    df_sorted = df.sort_values(["user_idx", "timestamp"])
    secuencias = (df_sorted.groupby("user_idx")["item_idx"]
                  .apply(list).to_dict())
    return secuencias


# ------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------

def procesar(nombre, loader_fn, min_interacciones):
    set_seed(SEED)
    print(f"[{nombre}] cargando interacciones...")
    df = loader_fn()
    print(f"[{nombre}] {len(df):,} interacciones crudas | "
          f"{df['user_id'].nunique():,} usuarios | "
          f"{df['item_id'].nunique():,} items")

    df = filtrar_min_interacciones(df, min_interacciones)
    print(f"[{nombre}] tras filtro <{min_interacciones} interacciones: "
          f"{len(df):,} interacciones | {df['user_id'].nunique():,} usuarios "
          f"| {df['item_id'].nunique():,} items")

    df, user2idx, item2idx = codificar_ids(df)
    n_users, n_items = len(user2idx), len(item2idx)

    edge_index = construir_grafo(df, n_users)
    secuencias = construir_secuencias(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/{nombre}_interacciones.csv", index=False)
    np.save(f"{OUT_DIR}/{nombre}_edge_index.npy", edge_index)
    with open(f"{OUT_DIR}/{nombre}_secuencias.json", "w") as f:
        json.dump({str(k): v for k, v in secuencias.items()}, f)
    with open(f"{OUT_DIR}/{nombre}_id_maps.json", "w") as f:
        json.dump({"n_users": n_users, "n_items": n_items}, f)

    print(f"[{nombre}] listo -> {n_users:,} usuarios, {n_items:,} items, "
          f"{edge_index.shape[1]:,} aristas (grafo no dirigido)\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"],
                         default="all")
    args = parser.parse_args()

    if args.dataset in ("olist", "all"):
        procesar("olist", cargar_interacciones_olist, MIN_INTERACTIONS_OLIST)
    if args.dataset in ("amazon", "all"):
        procesar("amazon", cargar_interacciones_amazon, MIN_INTERACTIONS_AMAZON)
