"""
modelo_fusion_5semillas.py
-----------------------------------------------------------------------
Reemplaza a modelo_hibrido.py y modelo_hibrido_gate.py para el
resultado final reportado (Seccion IV-D/IV-E, 5 semillas, seleccion de
epoca por NDCG@10 de validacion, evaluacion multi-objetivo restringida
al catalogo -- metricas_eval.py).

Diferencia clave de diseno frente a los scripts anteriores: NO
reentrena GNN y Transformer desde cero dentro de cada fusion (eso
significaria entrenar cada rama 3 veces mas, una por mecanismo de
fusion, con una sola semilla ademas). En vez de eso, REUSA:
  - data/processed/{nombre}_gnn_embeddings_seed{seed}.npy
    (embeddings de nodo ESTATICOS, calculados una sola vez por
    modelo_gnn_5semillas.py; no cambian durante la fusion)
  - data/processed/{nombre}_transformer_modelo_seed{seed}.pt
    (pesos COMPLETOS del Transformer, no solo item_emb, porque la
    representacion de usuario es el estado oculto DINAMICO en la
    ultima posicion real de su secuencia -- hay que poder correr
    forward() sobre el modelo ya entrenado, no basta un vector estatico)

y por default CONGELA ambas ramas (igual que los scripts anteriores):
solo se entrena la capa de fusion, que es barata (miles de parametros,
no millones).

Tres mecanismos (--fusion):
  - concat: proyeccion_usuario / proyeccion_item, Linear(emb_dim*2,
    fusion_dim) separados para usuario e item (igual que
    modelo_hibrido.py).
  - gate: CompuertaFusion por usuario y por item, peso escalar
    sigmoid(MLP(...)) en (0,1), fusionado = peso*gnn + (1-peso)*trans
    (igual que modelo_hibrido_gate.py).
  - suma: fusionado = 0.5*gnn + 0.5*trans, SIN parametros entrenables
    y SIN normalizacion de escala previa entre ramas -- limitacion ya
    documentada en el paper (ver ejecutar_suma_fija_segura.py,
    verificacion previa). No hay "entrenamiento" real: se evalua
    directo.

Entrada:
    data/processed/{nombre}_id_maps.json
    data/processed/{nombre}_train.csv / _val.csv / _test.csv
    data/processed/{nombre}_gnn_embeddings_seed{seed}.npy       (modelo_gnn_5semillas.py)
    data/processed/{nombre}_transformer_modelo_seed{seed}.pt    (modelo_transformer_5semillas.py)

Salida:
    results/{nombre}_fusion_{tipo}_5semillas.csv

Uso:
    python src/modelo_fusion_5semillas.py --dataset olist --fusion concat
    python src/modelo_fusion_5semillas.py --dataset olist --fusion gate
    python src/modelo_fusion_5semillas.py --dataset olist --fusion suma
    python src/modelo_fusion_5semillas.py --dataset olist --fusion all
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

from modelo_transformer import SecuenciaTransformer, secuencia_a_tensor, MAX_LEN
from modelo_transformer_5semillas import cargar_secuencias_train
from metricas_eval import construir_catalogo, evaluar_scores

PROC_DIR = "data/processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------------------
# Mecanismos de fusion
# ------------------------------------------------------------------

class FusionConcat(nn.Module):
    def __init__(self, emb_dim, fusion_dim):
        super().__init__()
        self.proy_usuario = nn.Linear(emb_dim * 2, fusion_dim)
        self.proy_item = nn.Linear(emb_dim * 2, fusion_dim)

    def usuario(self, gnn_u, trans_u):
        return self.proy_usuario(torch.cat([gnn_u, trans_u], dim=-1))

    def item(self, gnn_i, trans_i):
        return self.proy_item(torch.cat([gnn_i, trans_i], dim=-1))

    def peso_info(self):
        return {}


class CompuertaFusion(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim // 2), nn.ReLU(),
            nn.Linear(emb_dim // 2, 1), nn.Sigmoid())

    def forward(self, a, b):
        peso = self.red(torch.cat([a, b], dim=-1))
        return peso * a + (1.0 - peso) * b, peso.squeeze(-1)


class FusionGate(nn.Module):
    def __init__(self, emb_dim, fusion_dim=None):
        super().__init__()
        self.gate_usuario = CompuertaFusion(emb_dim)
        self.gate_item = CompuertaFusion(emb_dim)
        self._ultimo_peso_u, self._ultimo_peso_i = None, None

    def usuario(self, gnn_u, trans_u):
        f, peso = self.gate_usuario(gnn_u, trans_u)
        self._ultimo_peso_u = peso.detach()
        return f

    def item(self, gnn_i, trans_i):
        f, peso = self.gate_item(gnn_i, trans_i)
        self._ultimo_peso_i = peso.detach()
        return f

    def peso_info(self):
        info = {}
        if self._ultimo_peso_u is not None:
            info["peso_gate_usuario_promedio_hacia_gnn"] = float(self._ultimo_peso_u.mean())
        if self._ultimo_peso_i is not None:
            info["peso_gate_item_promedio_hacia_gnn"] = float(self._ultimo_peso_i.mean())
        return info


class FusionSuma(nn.Module):
    """Sin parametros entrenables, sin normalizacion de escala previa
    entre ramas (limitacion documentada)."""

    def __init__(self, emb_dim=None, fusion_dim=None):
        super().__init__()

    def usuario(self, gnn_u, trans_u):
        return 0.5 * gnn_u + 0.5 * trans_u

    def item(self, gnn_i, trans_i):
        return 0.5 * gnn_i + 0.5 * trans_i

    def peso_info(self):
        return {}


FUSIONES = {"concat": FusionConcat, "gate": FusionGate, "suma": FusionSuma}
ENTRENABLE = {"concat": True, "gate": True, "suma": False}


def bpr_loss(u_emb, pos_emb, neg_emb):
    pos_scores = (u_emb * pos_emb).sum(-1)
    neg_scores = (u_emb * neg_emb).sum(-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


# ------------------------------------------------------------------
# Carga de ramas ya entrenadas (NO se reentrenan)
# ------------------------------------------------------------------

def cargar_ramas(nombre, seed, proc_dir=PROC_DIR):
    idmaps = json.load(open(f"{proc_dir}/{nombre}_id_maps.json"))
    n_users, n_items = idmaps["n_users"], idmaps["n_items"]
    emb_dim = 64

    gnn_node_emb_np = np.load(f"{proc_dir}/{nombre}_gnn_embeddings_seed{seed}.npy")
    gnn_node_emb = torch.tensor(gnn_node_emb_np, dtype=torch.float32, device=DEVICE)

    trans_model = SecuenciaTransformer(n_items, max_len=MAX_LEN, emb_dim=emb_dim).to(DEVICE)
    estado = torch.load(f"{proc_dir}/{nombre}_transformer_modelo_seed{seed}.pt",
                         map_location=DEVICE)
    trans_model.load_state_dict(estado)
    trans_model.eval()
    for p in trans_model.parameters():
        p.requires_grad = False

    return idmaps, gnn_node_emb, trans_model, n_users, n_items, emb_dim


def estado_transformer_usuario(trans_model, secuencias_dict, u):
    """Estado oculto del Transformer en la ultima posicion real de la
    secuencia de train del usuario u (o None si no tiene secuencia)."""
    seq = secuencias_dict.get(u)
    if not seq:
        return None
    seq_tensor, longitud_real = secuencia_a_tensor(seq)
    entrada = torch.tensor([seq_tensor], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        h = trans_model(entrada)[0, longitud_real - 1]
    return h


# ------------------------------------------------------------------
# Entrenamiento de la capa de fusion (ramas congeladas)
# ------------------------------------------------------------------

def entrenar_fusion(nombre, tipo, seed, epochs=30, fusion_dim=64, lr=0.001,
                     batch_size=128, patience=5, k=10, proc_dir=PROC_DIR):
    set_seed(seed)
    idmaps, gnn_node_emb, trans_model, n_users, n_items, emb_dim = \
        cargar_ramas(nombre, seed, proc_dir)

    train = pd.read_csv(f"{proc_dir}/{nombre}_train.csv")
    val = pd.read_csv(f"{proc_dir}/{nombre}_val.csv")
    test = pd.read_csv(f"{proc_dir}/{nombre}_test.csv")
    secuencias_train = cargar_secuencias_train(train)
    secuencias_entrenables = {u: s for u, s in secuencias_train.items() if len(s) >= 2}
    usuarios = list(secuencias_entrenables.keys())

    fusion = FUSIONES[tipo](emb_dim, fusion_dim).to(DEVICE)

    if not ENTRENABLE[tipo]:
        # suma fija: sin parametros, no hay nada que optimizar.
        print(f"[{nombre}][{tipo}] sin parametros entrenables, evaluando directo.")
        return fusion, idmaps, gnn_node_emb, trans_model, train, val, test, 0, None

    optimizer = torch.optim.Adam(fusion.parameters(), lr=lr)
    print(f"[{nombre}][{tipo}] seed={seed}: entrenando fusion sobre "
          f"{len(usuarios):,} usuarios ({epochs} epocas max, paciencia {patience})")

    best_score, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        fusion.train()
        np.random.shuffle(usuarios)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(usuarios), batch_size):
            batch_users = usuarios[start:start + batch_size]
            u_validos, pos_validos, neg_validos, h_validos = [], [], [], []
            for u in batch_users:
                seq = secuencias_entrenables[u]
                entrada_t, long_real = secuencia_a_tensor(seq[:-1])
                with torch.no_grad():
                    h_seq = trans_model(torch.tensor([entrada_t], dtype=torch.long, device=DEVICE))[0]
                objetivo_t, _ = secuencia_a_tensor(seq[1:])
                for pos in range(long_real):
                    item_obj = objetivo_t[pos]
                    if item_obj == 0:
                        continue
                    u_validos.append(u)
                    pos_validos.append(item_obj - 1)
                    h_validos.append(h_seq[pos])
                    while True:
                        neg = np.random.randint(0, n_items)
                        if neg not in seq:
                            neg_validos.append(neg)
                            break
            if not u_validos:
                continue

            u_idx_t = torch.tensor(u_validos, dtype=torch.long, device=DEVICE)
            pos_idx_t = torch.tensor(pos_validos, dtype=torch.long, device=DEVICE)
            neg_idx_t = torch.tensor(neg_validos, dtype=torch.long, device=DEVICE)
            h_t = torch.stack(h_validos)

            gnn_u = gnn_node_emb[u_idx_t]
            fused_u = fusion.usuario(gnn_u, h_t)

            gnn_pos = gnn_node_emb[n_users + pos_idx_t]
            trans_pos = trans_model.item_emb(pos_idx_t + 1)
            fused_pos = fusion.item(gnn_pos, trans_pos)

            gnn_neg = gnn_node_emb[n_users + neg_idx_t]
            trans_neg = trans_model.item_emb(neg_idx_t + 1)
            fused_neg = fusion.item(gnn_neg, trans_neg)

            loss = bpr_loss(fused_u, fused_pos, fused_neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_metrics = evaluar(fusion, idmaps, gnn_node_emb, trans_model, train, val, k=k)
        score = val_metrics["secundario"][f"ndcg@{k}"]
        if score > best_score + 1e-12:
            best_score, best_state, best_epoch, stale = score, copy.deepcopy(fusion.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == epochs:
            print(f"[{nombre}][{tipo}] seed={seed} epoca {epoch}/{epochs} - "
                  f"loss BPR promedio: {total_loss / max(1, n_batches):.4f}")
        if stale >= patience:
            print(f"[{nombre}][{tipo}] seed={seed} parada temprana en epoca {epoch}; "
                  f"mejor epoca {best_epoch}")
            break

    fusion.load_state_dict(best_state)
    return fusion, idmaps, gnn_node_emb, trans_model, train, val, test, best_epoch, best_score


# ------------------------------------------------------------------
# Evaluacion: multi-objetivo, restringida al catalogo (metricas_eval.py)
# ------------------------------------------------------------------

@torch.no_grad()
def evaluar(fusion, idmaps, gnn_node_emb, trans_model, train, eval_df, k=10):
    fusion.eval()
    n_users, n_items = idmaps["n_users"], idmaps["n_items"]
    secuencias_train = cargar_secuencias_train(train)
    catalogo = construir_catalogo(train)

    todos_items = torch.arange(n_items, dtype=torch.long, device=DEVICE)
    gnn_items = gnn_node_emb[n_users + todos_items]
    trans_items = trans_model.item_emb(todos_items + 1)
    fused_item_all = fusion.item(gnn_items, trans_items)  # (n_items, fusion_dim)

    usuarios_a_evaluar = eval_df["user_idx"].unique()
    scores_por_usuario = {}
    for u in usuarios_a_evaluar:
        u = int(u)
        h = estado_transformer_usuario(trans_model, secuencias_train, u)
        if h is None:
            continue
        gnn_u = gnn_node_emb[u].unsqueeze(0)
        fused_u = fusion.usuario(gnn_u, h.unsqueeze(0))[0]
        scores_por_usuario[u] = (fused_item_all @ fused_u).cpu().numpy()

    return evaluar_scores(scores_por_usuario, train, eval_df, catalogo, k=k)


# ------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------

def procesar(nombre, tipo, epochs=30, k=10, seeds=None, patience=5, proc_dir=PROC_DIR):
    seeds = seeds or [42, 123, 2024, 2025, 2026]
    os.makedirs("results", exist_ok=True)
    rows = []
    for seed in seeds:
        fusion, idmaps, gnn_node_emb, trans_model, train, val, test, best_epoch, val_score = \
            entrenar_fusion(nombre, tipo, seed, epochs=epochs, patience=patience, k=k,
                             proc_dir=proc_dir)
        metricas = evaluar(fusion, idmaps, gnn_node_emb, trans_model, train, test, k=k)
        fila = {"dataset": nombre, "fusion": tipo, "seed": seed, "best_epoch": best_epoch,
                "val_ndcg": val_score}
        fila.update(fusion.peso_info())
        for analisis in ("principal", "secundario"):
            for clave, valor in metricas[analisis].items():
                fila[f"{analisis}_{clave}"] = valor
        rows.append(fila)
        print(f"[{nombre}][{tipo}] seed={seed} test={metricas}")

    output = f"results/{nombre}_fusion_{tipo}_5semillas.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"[{nombre}][{tipo}] resultados guardados: {output}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"], default="all")
    parser.add_argument("--fusion", choices=["concat", "gate", "suma", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=20,
                         help="20 epocas / paciencia 5, igual que declara IV-E "
                              "('ajuste equitativo') para las fusiones")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    tipos = list(FUSIONES.keys()) if args.fusion == "all" else [args.fusion]
    for nombre in (["olist", "amazon"] if args.dataset == "all" else [args.dataset]):
        for tipo in tipos:
            procesar(nombre, tipo, epochs=args.epochs, k=args.k, patience=args.patience)
