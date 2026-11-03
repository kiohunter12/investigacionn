"""
reconstruir_particion_olist.py
-----------------------------------------------------------------------
Reconstruccion del subconjunto Olist y particion sin fuga, por PERSONA
y por MOMENTO DE COMPRA (no por fila individual).

Reemplaza la logica de preprocesar_dataset.py + particionar_dataset.py
para Olist, que particionaba por fila y por eso permitia que productos
del mismo pedido (misma marca temporal) quedaran repartidos entre
validacion y prueba (ver C1 / hallazgo de fuga: 101/138 casos).

Reglas (verificadas manualmente contra las tablas crudas, coinciden
exacto con el Cuadro de conteos del short_paper_P1_v2):
  1. usuario = customer_unique_id (persona, no pedido)
  2. interaccion = compra de un producto (product_id) con marca
     temporal order_purchase_timestamp
  3. se excluyen pedidos "canceled"/"unavailable" y duplicados
     persona-producto-momento
  4. "momento de compra" = marca temporal exacta (agrupa productos del
     mismo pedido)
  5. elegibles = personas con >=2 momentos de compra distintos
  6. el ULTIMO momento de una persona elegible se reserva como
     objetivo; sus productos "nuevos" (no comprados antes por esa
     persona) son los objetivos evaluables; los "repetidos" se excluyen
  7. las personas con objetivo se dividen 50/50 (semilla 42) en
     validacion y prueba, por PERSONA (nunca por fila), así ningun
     pedido ni ninguna persona queda partido entre particiones
  8. entrenamiento = todo lo que no sea el ultimo momento de una
     persona elegible (incluye integras a las personas con 1 solo
     momento)

Salida (data/processed/):
    olist_train_v2.csv, olist_val_v2.csv, olist_test_v2.csv
        columnas: user_id, item_id, rating, timestamp, user_idx, item_idx
    olist_id_maps_v2.json   {"n_users":..., "n_items":...}
    olist_conteos_v2.csv    conteos por paso, para verificar contra el paper

Uso:
    python reconstruir_particion_olist.py
-----------------------------------------------------------------------
"""
import json
import os

import numpy as np
import pandas as pd

SEED = 42
RAW_DIR = "data/raw"
PROC_DIR = "data/processed"


def cargar_interacciones_olist(raw_dir=RAW_DIR):
    """Identico a preprocesar_dataset.cargar_interacciones_olist:
    customer_unique_id como usuario, excluye canceled/unavailable,
    dedup por (user_id, item_id, timestamp)."""
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


def construir_particion(df):
    conteos = []

    def registrar(paso, d):
        conteos.append({
            "paso": paso, "filas": len(d),
            "personas": d["user_id"].nunique(),
            "productos": d["item_id"].nunique(),
        })

    registrar("3_sin_duplicados", df)

    momentos_por_persona = df.groupby("user_id")["timestamp"].nunique()
    elegibles = set(momentos_por_persona[momentos_por_persona >= 2].index)
    print(f"personas elegibles (>=2 momentos): {len(elegibles):,}")

    df_elig = df[df["user_id"].isin(elegibles)]
    ultimo_momento = df_elig.groupby("user_id")["timestamp"].transform("max")
    es_ultimo = df_elig["timestamp"] == ultimo_momento
    filas_ultimo = df_elig[es_ultimo].copy()
    historial_previo = df_elig[~es_ultimo]

    prev_sets = historial_previo.groupby("user_id")["item_id"].apply(set).to_dict()
    filas_ultimo["nuevo"] = filas_ultimo.apply(
        lambda r: r["item_id"] not in prev_sets.get(r["user_id"], set()), axis=1)
    objetivos_nuevos = filas_ultimo[filas_ultimo["nuevo"]].drop(columns=["nuevo"])
    objetivos_repetidos = filas_ultimo[~filas_ultimo["nuevo"]]
    print(f"objetivos nuevos: {len(objetivos_nuevos):,} filas, "
          f"{objetivos_nuevos['user_id'].nunique():,} personas")
    print(f"objetivos repetidos (excluidos): {len(objetivos_repetidos):,}")

    no_elegibles = df[~df["user_id"].isin(elegibles)]
    train_df = pd.concat([no_elegibles, historial_previo]).reset_index(drop=True)
    registrar("4_entrenamiento", train_df)
    registrar("5_objetivos_nuevos", objetivos_nuevos)

    personas_objetivo = sorted(objetivos_nuevos["user_id"].unique().tolist())
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(personas_objetivo))
    personas_shuffled = [personas_objetivo[i] for i in perm]
    mitad = len(personas_shuffled) // 2
    personas_val = set(personas_shuffled[:mitad])
    personas_test = set(personas_shuffled[mitad:])

    val_df = objetivos_nuevos[objetivos_nuevos["user_id"].isin(personas_val)].reset_index(drop=True)
    test_df = objetivos_nuevos[objetivos_nuevos["user_id"].isin(personas_test)].reset_index(drop=True)
    registrar("6a_validacion", val_df)
    registrar("6b_prueba", test_df)

    assert set(val_df["user_id"]).isdisjoint(set(test_df["user_id"])), \
        "FUGA: una persona quedo en validacion y en prueba a la vez"
    assert len(personas_val) + len(personas_test) == len(personas_objetivo)

    return train_df, val_df, test_df, pd.DataFrame(conteos)


def codificar_ids(train_df, val_df, test_df):
    todos_usuarios = pd.concat([train_df["user_id"], val_df["user_id"],
                                 test_df["user_id"]]).unique()
    todos_items = pd.concat([train_df["item_id"], val_df["item_id"],
                              test_df["item_id"]]).unique()
    user2idx = {u: i for i, u in enumerate(sorted(todos_usuarios))}
    item2idx = {p: i for i, p in enumerate(sorted(todos_items))}

    for d in (train_df, val_df, test_df):
        d["user_idx"] = d["user_id"].map(user2idx)
        d["item_idx"] = d["item_id"].map(item2idx)
    return train_df, val_df, test_df, len(user2idx), len(item2idx)


def verificar_sin_fuga(train_df, val_df, test_df):
    train_last = train_df.groupby("user_id")["timestamp"].max()
    for nombre, d in [("val", val_df), ("test", test_df)]:
        m = d.merge(train_last.rename("ultimo_train"), on="user_id", how="left")
        viola = m[m["ultimo_train"] >= m["timestamp"]]
        assert len(viola) == 0, f"FUGA TEMPORAL en {nombre}: {len(viola)} filas"
    assert set(val_df["user_id"]).isdisjoint(set(test_df["user_id"]))
    print("Verificacion sin fuga: OK (las 3 condiciones se cumplen)")


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    print("[olist] cargando y deduplicando interacciones...")
    df = cargar_interacciones_olist()
    print(f"[olist] tras cruce+exclusion+dedup: {len(df):,} filas | "
          f"{df['user_id'].nunique():,} personas | {df['item_id'].nunique():,} productos")

    train_df, val_df, test_df, conteos = construir_particion(df)
    train_df, val_df, test_df, n_users, n_items = codificar_ids(train_df, val_df, test_df)
    verificar_sin_fuga(train_df, val_df, test_df)

    train_df.to_csv(f"{PROC_DIR}/olist_train_v2.csv", index=False)
    val_df.to_csv(f"{PROC_DIR}/olist_val_v2.csv", index=False)
    test_df.to_csv(f"{PROC_DIR}/olist_test_v2.csv", index=False)
    with open(f"{PROC_DIR}/olist_id_maps_v2.json", "w") as f:
        json.dump({"n_users": n_users, "n_items": n_items}, f)
    conteos.to_csv(f"{PROC_DIR}/olist_conteos_v2.csv", index=False)

    print(f"\n[olist] train={len(train_df):,} val={len(val_df):,} "
          f"test={len(test_df):,} | n_users={n_users:,} n_items={n_items:,}")
    print("\nConteos por paso:")
    print(conteos.to_string(index=False))


if __name__ == "__main__":
    main()
