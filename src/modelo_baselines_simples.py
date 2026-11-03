"""
modelo_baselines_simples.py
-----------------------------------------------------------------------
Las 3 lineas base NO neuronales exigidas en la Seccion IV-E del short
paper (OE3): "Popularidad (numero de compras de cada producto en
entrenamiento), ItemKNN con similitud coseno entre productos y BPR-MF,
factorizacion matricial optimizada con BPR." No existian en el
proyecto (src/modelo_baselines.py solo tiene TransGNN y SIGformer, que
son lineas base de GRAFO de la literatura, no estas).

Las tres se evaluan con el MISMO protocolo que las ramas (metricas_eval.py):
multi-objetivo, restringido al catalogo de entrenamiento, analisis
principal + secundario, mismas 5 semillas (aunque Popularidad e
ItemKNN no son estocasticos: su resultado no cambia entre semillas,
solo se repiten para que la tabla final tenga una fila por semilla y
por configuracion, igual que las demas).

Modelos:
  1. Popularidad: score(i) = numero de compras de i en train (igual
     para toda persona). No tiene hiperparametros ni semilla.
  2. ItemKNN: similitud coseno entre columnas de la matriz
     usuario-item (binaria) de train; score(u,i) = suma de similitudes
     entre i y los items ya comprados por u, restringido a los
     n_vecinos mas similares a cada item (ver IV-E, "ajuste equitativo":
     n_vecinos en {20,50,100}).
  3. BPR-MF: factorizacion matricial (embeddings de usuario e item,
     sin grafo ni secuencia) optimizada con la misma perdida BPR que
     las demas ramas.

Entrada:
    data/processed/{nombre}_train.csv / _val.csv / _test.csv
    data/processed/{nombre}_id_maps.json

Salida:
    results/{nombre}_baselines_simples.csv

Uso:
    python src/modelo_baselines_simples.py --dataset olist
-----------------------------------------------------------------------
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from metricas_eval import construir_catalogo, evaluar_scores

PROC_DIR = "data/processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def cargar_datos(nombre, proc_dir=PROC_DIR):
    idmaps = json.load(open(f"{proc_dir}/{nombre}_id_maps.json"))
    train = pd.read_csv(f"{proc_dir}/{nombre}_train.csv")
    val = pd.read_csv(f"{proc_dir}/{nombre}_val.csv")
    test = pd.read_csv(f"{proc_dir}/{nombre}_test.csv")
    return idmaps, train, val, test


# ------------------------------------------------------------------
# 1. Popularidad
# ------------------------------------------------------------------

def scores_popularidad(train, n_items):
    conteo = train["item_idx"].value_counts()
    scores = np.zeros(n_items, dtype=np.float64)
    scores[conteo.index.to_numpy()] = conteo.to_numpy()
    return scores  # mismo vector para toda persona


# ------------------------------------------------------------------
# 2. ItemKNN (similitud coseno item-item sobre la matriz binaria)
# ------------------------------------------------------------------

def construir_matriz_usuario_item(train, n_users, n_items):
    filas = train["user_idx"].to_numpy()
    cols = train["item_idx"].to_numpy()
    datos = np.ones(len(train), dtype=np.float64)
    m = sparse.csr_matrix((datos, (filas, cols)), shape=(n_users, n_items))
    m.data[:] = 1.0  # binaria, por si hay compras repetidas del mismo item
    return m


def similitud_coseno_items(matriz_ui, n_vecinos=50):
    """Similitud coseno entre columnas (items). Devuelve, para cada item,
    solo sus `n_vecinos` mas similares (el resto en 0) para que el score
    sea manejable en memoria con ~33k items."""
    norm = sparse.linalg.norm(matriz_ui, axis=0)
    norm[norm == 0] = 1e-8
    matriz_norm = matriz_ui.multiply(1.0 / norm)
    sim = (matriz_norm.T @ matriz_norm).tocsr()  # (n_items, n_items), coseno
    sim.setdiag(0)

    n_items = sim.shape[0]
    filas_top, cols_top, datos_top = [], [], []
    for i in range(n_items):
        inicio, fin = sim.indptr[i], sim.indptr[i + 1]
        if fin == inicio:
            continue
        vecinos = sim.indices[inicio:fin]
        valores = sim.data[inicio:fin]
        if len(valores) > n_vecinos:
            top = np.argpartition(-valores, n_vecinos - 1)[:n_vecinos]
            vecinos, valores = vecinos[top], valores[top]
        filas_top.extend([i] * len(vecinos))
        cols_top.extend(vecinos)
        datos_top.extend(valores)
    sim_top = sparse.csr_matrix((datos_top, (filas_top, cols_top)),
                                 shape=(n_items, n_items))
    return sim_top


def scores_itemknn_usuarios(matriz_ui, sim_top, usuarios_idx):
    """score(u, :) = suma de similitudes de los items comprados por u con
    todos los demas items, SOLO para los usuarios pedidos (evita
    materializar la matriz densa completa n_users x n_items, que para
    Olist pesaria ~23 GiB en float64). Devuelve {user_idx: vector (n_items,)}."""
    usuarios_idx = np.asarray(sorted(set(int(u) for u in usuarios_idx)))
    sub = matriz_ui[usuarios_idx] @ sim_top  # (len(usuarios_idx), n_items), sparse
    sub = sub.toarray()
    return {int(u): sub[i] for i, u in enumerate(usuarios_idx)}


# ------------------------------------------------------------------
# 3. BPR-MF (factorizacion matricial, sin grafo ni secuencia)
# ------------------------------------------------------------------

class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(-1)


def bpr_loss(pos_scores, neg_scores):
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def entrenar_bprmf(nombre, train, val, idmaps, emb_dim=64, lr=0.01,
                    batch_size=1024, epochs=50, patience=5, seed=42, k=10):
    set_seed(seed)
    n_users, n_items = idmaps["n_users"], idmaps["n_items"]
    model = BPRMF(n_users, n_items, emb_dim=emb_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    items_por_usuario = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    usuarios_train = train["user_idx"].to_numpy()
    items_train = train["item_idx"].to_numpy()
    n_train = len(train)
    catalogo = construir_catalogo(train)

    best_score, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            u_batch = usuarios_train[idx]
            pos_batch = items_train[idx]
            neg_batch = np.empty(len(u_batch), dtype=np.int64)
            for j, u in enumerate(u_batch):
                vistos = items_por_usuario[u]
                while True:
                    cand = np.random.randint(0, n_items)
                    if cand not in vistos:
                        neg_batch[j] = cand
                        break

            u_t = torch.tensor(u_batch, dtype=torch.long, device=DEVICE)
            pos_t = torch.tensor(pos_batch, dtype=torch.long, device=DEVICE)
            neg_t = torch.tensor(neg_batch, dtype=torch.long, device=DEVICE)

            pos_scores = model(u_t, pos_t)
            neg_scores = model(u_t, neg_t)
            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        with torch.no_grad():
            model.eval()
            user_emb = model.user_emb.weight.cpu().numpy()
            item_emb = model.item_emb.weight.cpu().numpy()
            scores_val = {int(u): user_emb[int(u)] @ item_emb.T
                          for u in val["user_idx"].unique()}
            val_metrics = evaluar_scores(scores_val, train, val, catalogo, k=k)
            score = val_metrics["secundario"][f"ndcg@{k}"]

        if score > best_score + 1e-12:
            best_score, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_epoch, best_score


# ------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------

def evaluar_matriz_scores(scores_matrix_or_vector, idmaps, train, eval_df, catalogo,
                           k=10, es_vector_global=False):
    usuarios = eval_df["user_idx"].unique()
    if es_vector_global:
        scores_por_usuario = {int(u): scores_matrix_or_vector for u in usuarios}
    else:
        scores_por_usuario = {int(u): scores_matrix_or_vector[int(u)] for u in usuarios}
    return evaluar_scores(scores_por_usuario, train, eval_df, catalogo, k=k)


def procesar(nombre, k=10, seeds=None, epochs_bprmf=50, patience=5,
             n_vecinos=50, proc_dir=PROC_DIR):
    seeds = seeds or [42, 123, 2024, 2025, 2026]
    os.makedirs("results", exist_ok=True)
    idmaps, train, val, test = cargar_datos(nombre, proc_dir)
    n_users, n_items = idmaps["n_users"], idmaps["n_items"]
    catalogo = construir_catalogo(train)

    rows = []

    # --- Popularidad (deterministico: una sola vez, se repite por semilla) ---
    print(f"[{nombre}] Popularidad...")
    pop_scores = scores_popularidad(train, n_items)
    metricas_pop = evaluar_matriz_scores(pop_scores, idmaps, train, test, catalogo,
                                          k=k, es_vector_global=True)
    for seed in seeds:
        fila = {"dataset": nombre, "modelo": "popularidad", "seed": seed, "best_epoch": None}
        for analisis in ("principal", "secundario"):
            for clave, valor in metricas_pop[analisis].items():
                fila[f"{analisis}_{clave}"] = valor
        rows.append(fila)
    print(f"[{nombre}] Popularidad test (secundario) = {metricas_pop['secundario']}")

    # --- ItemKNN (deterministico) ---
    print(f"[{nombre}] ItemKNN (n_vecinos={n_vecinos})...")
    matriz_ui = construir_matriz_usuario_item(train, n_users, n_items)
    sim_top = similitud_coseno_items(matriz_ui, n_vecinos=n_vecinos)
    scores_knn = scores_itemknn_usuarios(matriz_ui, sim_top, test["user_idx"].unique())
    metricas_knn = evaluar_scores(scores_knn, train, test, catalogo, k=k)
    for seed in seeds:
        fila = {"dataset": nombre, "modelo": "itemknn", "seed": seed, "best_epoch": None,
                "n_vecinos": n_vecinos}
        for analisis in ("principal", "secundario"):
            for clave, valor in metricas_knn[analisis].items():
                fila[f"{analisis}_{clave}"] = valor
        rows.append(fila)
    print(f"[{nombre}] ItemKNN test (secundario) = {metricas_knn['secundario']}")

    # --- BPR-MF (estocastico: 5 semillas reales) ---
    for seed in seeds:
        print(f"[{nombre}] BPR-MF seed={seed}...")
        model, best_epoch, val_ndcg = entrenar_bprmf(
            nombre, train, val, idmaps, epochs=epochs_bprmf, patience=patience,
            seed=seed, k=k)
        with torch.no_grad():
            user_emb = model.user_emb.weight.cpu().numpy()
            item_emb = model.item_emb.weight.cpu().numpy()
        scores_bprmf = {int(u): user_emb[int(u)] @ item_emb.T
                         for u in test["user_idx"].unique()}
        metricas = evaluar_scores(scores_bprmf, train, test, catalogo, k=k)
        fila = {"dataset": nombre, "modelo": "bprmf", "seed": seed, "best_epoch": best_epoch,
                "val_ndcg": val_ndcg}
        for analisis in ("principal", "secundario"):
            for clave, valor in metricas[analisis].items():
                fila[f"{analisis}_{clave}"] = valor
        rows.append(fila)
        print(f"[{nombre}] BPR-MF seed={seed} test={metricas}")

    output = f"results/{nombre}_baselines_simples.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"[{nombre}] resultados guardados: {output}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"], default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epochs_bprmf", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n_vecinos", type=int, default=50)
    args = parser.parse_args()
    for nombre in (["olist", "amazon"] if args.dataset == "all" else [args.dataset]):
        procesar(nombre, k=args.k, epochs_bprmf=args.epochs_bprmf,
                  patience=args.patience, n_vecinos=args.n_vecinos)
