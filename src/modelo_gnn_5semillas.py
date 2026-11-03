"""
modelo_gnn.py
-----------------------------------------------------------------------
Rama GNN del modelo hibrido (Semanas 3-4 del cronograma).

Encoder de grafo usuario-item con atencion (GAT), adaptando el
principio de atencion sobre grafos de GraphMind (idea base del PPI).
Entrena con perdida BPR (Bayesian Personalized Ranking) sobre el grafo
bipartito ya construido por preprocesar_dataset.py, y evalua con
Recall@K / NDCG@K sobre la particion leave-last-out (data/processed/*_test.csv).

Entrada (generada por el pipeline de datos, Hito 1):
    data/processed/{nombre}_edge_index.npy
    data/processed/{nombre}_id_maps.json
    data/processed/{nombre}_train.csv / _val.csv / _test.csv

Salida:
    data/processed/{nombre}_gnn_embeddings.npy   (embeddings finales, un vector por nodo)
    Metricas Recall@10 / NDCG@10 impresas en consola

Uso (desde /workspace dentro del contenedor Docker):
    python src/modelo_gnn.py --dataset olist
    python src/modelo_gnn.py --dataset amazon
    python src/modelo_gnn.py --dataset all
-----------------------------------------------------------------------
"""
import argparse
import json
import os
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from metricas_eval import construir_catalogo, evaluar_scores

SEED = 42
PROC_DIR = "data/processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------------------
# Modelo: encoder GAT sobre el grafo bipartito usuario-item
# ------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """Embeddings de nodo (usuarios + items comparten un unico espacio,
    ver construir_grafo() en preprocesar_dataset.py) refinados con
    capas de atencion sobre el grafo (GATConv)."""

    def __init__(self, n_nodes, emb_dim=64, n_layers=2, heads=2):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GATConv(emb_dim, emb_dim // heads, heads=heads))

    def forward(self, edge_index):
        x = self.embedding.weight
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        return x  # (n_nodes, emb_dim) -- embedding final de cada nodo


def bpr_loss(u_emb, pos_emb, neg_emb):
    """Bayesian Personalized Ranking: empuja el score del item positivo
    por encima del negativo para el mismo usuario."""
    pos_scores = (u_emb * pos_emb).sum(-1)
    neg_scores = (u_emb * neg_emb).sum(-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


# ------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------

def cargar_datos(nombre, proc_dir=PROC_DIR):
    idmaps = json.load(open(f"{proc_dir}/{nombre}_id_maps.json"))
    edge_index = np.load(f"{proc_dir}/{nombre}_edge_index.npy")
    train = pd.read_csv(f"{proc_dir}/{nombre}_train.csv")
    val = pd.read_csv(f"{proc_dir}/{nombre}_val.csv")
    test = pd.read_csv(f"{proc_dir}/{nombre}_test.csv")
    return idmaps, edge_index, train, val, test


def muestrear_negativos(usuarios, n_items, items_vistos_por_usuario):
    """Un item negativo (no interactuado) por cada interaccion positiva."""
    negativos = np.empty(len(usuarios), dtype=np.int64)
    for i, u in enumerate(usuarios):
        vistos = items_vistos_por_usuario[u]
        while True:
            cand = np.random.randint(0, n_items)
            if cand not in vistos:
                negativos[i] = cand
                break
    return negativos


# ------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------

def entrenar(nombre, epochs=20, emb_dim=64, lr=0.01, batch_size=1024, seed=42, patience=8, k=10):
    set_seed(seed)
    idmaps, edge_index_np, train, val, test = cargar_datos(nombre)
    n_users, n_items = idmaps["n_users"], idmaps["n_items"]
    n_nodes = n_users + n_items

    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=DEVICE)
    model = GNNEncoder(n_nodes, emb_dim=emb_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # items ya vistos por cada usuario (para no muestrear negativos falsos)
    items_por_usuario = (train.groupby("user_idx")["item_idx"]
                          .apply(set).to_dict())
    for u in items_por_usuario:
        items_por_usuario[u] = items_por_usuario[u] | set()  # copia

    usuarios_train = train["user_idx"].to_numpy()
    items_train = train["item_idx"].to_numpy()
    n_train = len(train)

    print(f"[{nombre}] entrenando GNN: {n_users:,} usuarios, {n_items:,} items, "
          f"{n_train:,} interacciones de train, {epochs} epocas")

    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            u_batch = usuarios_train[idx]
            pos_batch = items_train[idx]
            neg_batch = muestrear_negativos(u_batch, n_items, items_por_usuario)

            node_emb = model(edge_index)
            u_emb = node_emb[u_batch]
            pos_emb = node_emb[pos_batch + n_users]
            neg_emb = node_emb[neg_batch + n_users]

            loss = bpr_loss(u_emb, pos_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # La validación nunca utiliza las aristas de validación en el grafo.
        val_metrics = evaluar(model, edge_index, idmaps, train, val, k=k)
        # Selección de época por NDCG@10 de validación, análisis secundario
        # (todas las personas con objetivo; IV-D).
        score = val_metrics["secundario"][f"ndcg@{k}"]
        if score > best_score + 1e-12:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == epochs:
            print(f"[{nombre}] epoca {epoch}/{epochs} - loss BPR promedio: "
                  f"{total_loss / n_batches:.4f}")

        if stale >= patience:
            print(f"[{nombre}] parada temprana en época {epoch}; mejor época {best_epoch}")
            break
    model.load_state_dict(best_state)
    return model, edge_index, idmaps, train, val, test, best_epoch, best_score


# ------------------------------------------------------------------
# Evaluacion: Recall@K / HR@K / NDCG@K multi-objetivo, restringida al
# catalogo de entrenamiento (Seccion IV-D, ver metricas_eval.py)
# ------------------------------------------------------------------

@torch.no_grad()
def evaluar(model, edge_index, idmaps, train, test, k=10):
    model.eval()
    n_users, n_items = idmaps["n_users"], idmaps["n_items"]
    node_emb = model(edge_index)
    user_emb = node_emb[:n_users]
    item_emb = node_emb[n_users:]

    catalogo = construir_catalogo(train)
    usuarios_a_evaluar = test["user_idx"].unique()
    scores_por_usuario = {}
    for u in usuarios_a_evaluar:
        scores_por_usuario[int(u)] = (user_emb[int(u)] @ item_emb.T).cpu().numpy()

    return evaluar_scores(scores_por_usuario, train, test, catalogo, k=k)


# ------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------

def procesar(nombre, epochs=50, k=10, proc_dir=PROC_DIR, seeds=None, patience=8):
    seeds = seeds or [42, 123, 2024, 2025, 2026]
    os.makedirs("results", exist_ok=True)
    rows = []
    for seed in seeds:
        model, edge_index, idmaps, train, val, test, best_epoch, val_ndcg = entrenar(
            nombre, epochs=epochs, seed=seed, patience=patience, k=k)
        # Test solo tras seleccionar la mejor época en validación.
        metricas = evaluar(model, edge_index, idmaps, train, test, k=k)
        fila = {"dataset": nombre, "seed": seed, "best_epoch": best_epoch,
                "val_ndcg": val_ndcg}
        for analisis in ("principal", "secundario"):
            for clave, valor in metricas[analisis].items():
                fila[f"{analisis}_{clave}"] = valor
        rows.append(fila)
        print(f"[{nombre}] seed={seed} test={metricas}")
        with torch.no_grad():
            embeddings = model(edge_index).cpu().numpy()
        np.save(f"{proc_dir}/{nombre}_gnn_embeddings_seed{seed}.npy", embeddings)
    output = f"results/{nombre}_gnn_5semillas.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"[{nombre}] resultados guardados: {output}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    for nombre in (["olist", "amazon"] if args.dataset == "all" else [args.dataset]):
        procesar(nombre, epochs=args.epochs, k=args.k, patience=args.patience)
