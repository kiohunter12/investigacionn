"""
modelo_transformer_5semillas.py
-----------------------------------------------------------------------
Extiende modelo_transformer.py para que siga EXACTAMENTE el mismo
protocolo que modelo_gnn_5semillas.py (Seccion IV-D del short paper):
  - division train/val/test (antes esta rama solo usaba train/test,
    sin seleccion de epoca por validacion);
  - seleccion de la mejor epoca por NDCG@10 de VALIDACION, con paciencia;
  - 5 semillas {42,123,2024,2025,2026};
  - evaluacion multi-objetivo restringida al catalogo de entrenamiento
    (metricas_eval.py), igual que la rama GNN -- antes esta rama tambien
    tenia el mismo error de tomar un unico objetivo por persona.

No modifica modelo_transformer.py (se deja como referencia / script
exploratorio de una sola semilla); este es el que se usa para el
resultado final reportado en el paper, igual que modelo_gnn_5semillas.py
reemplazo a modelo_gnn.py para la rama GNN.

Entrada:
    data/processed/{nombre}_train.csv / _val.csv / _test.csv
    data/processed/{nombre}_id_maps.json

Salida:
    data/processed/{nombre}_transformer_embeddings_seed{seed}.npy
    results/{nombre}_transformer_5semillas.csv

Uso:
    python src/modelo_transformer_5semillas.py --dataset olist
-----------------------------------------------------------------------
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from modelo_transformer import (
    SecuenciaTransformer, bpr_loss, cargar_secuencias_train,
    secuencia_a_tensor, MAX_LEN, DEVICE,
)
from metricas_eval import construir_catalogo, evaluar_scores

PROC_DIR = "data/processed"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def entrenar(nombre, epochs=50, emb_dim=64, lr=0.001, batch_size=128,
             seed=42, patience=8, k=10, proc_dir=PROC_DIR):
    set_seed(seed)
    idmaps = json.load(open(f"{proc_dir}/{nombre}_id_maps.json"))
    n_items = idmaps["n_items"]

    train = pd.read_csv(f"{proc_dir}/{nombre}_train.csv")
    val = pd.read_csv(f"{proc_dir}/{nombre}_val.csv")
    test = pd.read_csv(f"{proc_dir}/{nombre}_test.csv")

    secuencias = cargar_secuencias_train(train)
    secuencias = {u: s for u, s in secuencias.items() if len(s) >= 2}
    usuarios = list(secuencias.keys())

    model = SecuenciaTransformer(n_items, max_len=MAX_LEN, emb_dim=emb_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[{nombre}] entrenando Transformer: {len(usuarios):,} usuarios con "
          f"secuencia util, {n_items:,} items, {epochs} epocas, seed={seed}")

    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(usuarios)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(usuarios), batch_size):
            batch_users = usuarios[start:start + batch_size]
            entradas, objetivos = [], []
            for u in batch_users:
                seq = secuencias[u]
                entradas.append(secuencia_a_tensor(seq[:-1])[0])
                objetivos.append(secuencia_a_tensor(seq[1:])[0])

            entradas = torch.tensor(entradas, dtype=torch.long, device=DEVICE)
            objetivos_t = torch.tensor(objetivos, dtype=torch.long, device=DEVICE)

            negativos = np.random.randint(1, n_items + 1, size=objetivos_t.shape)
            negativos_t = torch.tensor(negativos, dtype=torch.long, device=DEVICE)

            h = model(entradas)
            mask_validos = objetivos_t != 0

            pos_emb = model.item_emb(objetivos_t)
            neg_emb = model.item_emb(negativos_t)

            h_v = h[mask_validos]
            pos_v = pos_emb[mask_validos]
            neg_v = neg_emb[mask_validos]
            if h_v.shape[0] == 0:
                continue

            loss = bpr_loss(h_v, pos_v, neg_v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        val_metrics = evaluar(model, idmaps, train, val, k=k)
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
                  f"{total_loss / max(1, n_batches):.4f}")

        if stale >= patience:
            print(f"[{nombre}] parada temprana en epoca {epoch}; mejor epoca {best_epoch}")
            break

    model.load_state_dict(best_state)
    return model, idmaps, train, val, test, best_epoch, best_score


@torch.no_grad()
def evaluar(model, idmaps, train, eval_df, k=10):
    """Multi-objetivo, restringido al catalogo de train (metricas_eval.py)."""
    model.eval()
    secuencias_train = cargar_secuencias_train(train)
    catalogo = construir_catalogo(train)
    item_emb_all = model.item_emb.weight[1:]  # sin la fila de padding (0)

    usuarios_a_evaluar = eval_df["user_idx"].unique()
    scores_por_usuario = {}
    for u in usuarios_a_evaluar:
        u = int(u)
        seq = secuencias_train.get(u)
        if not seq:
            continue
        seq_tensor, longitud_real = secuencia_a_tensor(seq)
        entrada = torch.tensor([seq_tensor], dtype=torch.long, device=DEVICE)
        h = model(entrada)[0, longitud_real - 1]
        scores_por_usuario[u] = (item_emb_all @ h).cpu().numpy()

    return evaluar_scores(scores_por_usuario, train, eval_df, catalogo, k=k)


def procesar(nombre, epochs=50, k=10, proc_dir=PROC_DIR, seeds=None, patience=8):
    seeds = seeds or [42, 123, 2024, 2025, 2026]
    os.makedirs("results", exist_ok=True)
    rows = []
    for seed in seeds:
        model, idmaps, train, val, test, best_epoch, val_ndcg = entrenar(
            nombre, epochs=epochs, seed=seed, patience=patience, k=k, proc_dir=proc_dir)
        metricas = evaluar(model, idmaps, train, test, k=k)
        fila = {"dataset": nombre, "seed": seed, "best_epoch": best_epoch,
                "val_ndcg": val_ndcg}
        for analisis in ("principal", "secundario"):
            for clave, valor in metricas[analisis].items():
                fila[f"{analisis}_{clave}"] = valor
        rows.append(fila)
        print(f"[{nombre}] seed={seed} test={metricas}")

        with torch.no_grad():
            embeddings = model.item_emb.weight[1:].cpu().numpy()
        np.save(f"{proc_dir}/{nombre}_transformer_embeddings_seed{seed}.npy", embeddings)
        # Se guarda tambien el modelo completo (no solo item_emb): las
        # fusiones necesitan el estado oculto DINAMICO por usuario
        # (ultima posicion real de su secuencia), que no es un embedding
        # estatico y por eso no basta con el .npy de arriba.
        torch.save(model.state_dict(),
                   f"{proc_dir}/{nombre}_transformer_modelo_seed{seed}.pt")

    output = f"results/{nombre}_transformer_5semillas.csv"
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
