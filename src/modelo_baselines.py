"""
modelo_baselines.py
-----------------------------------------------------------------------
Baselines externos para la Tabla IV del short paper (Hito 3): LightGCN,
TransGNN y SIGformer, entrenados y evaluados con el MISMO protocolo que
las ramas propias (seed=42, leave-last-out, perdida BPR, Recall@K /
NDCG@K sobre el ranking completo de items) para que la comparacion sea
justa.

SASRec NO se reentrena aqui: la rama Transformer de este proyecto
(modelo_transformer.py) ya sigue el enfoque SASRec (autoatencion causal
+ prediccion del siguiente item + perdida BPR), asi que su resultado ya
obtenido en la Tabla II se reporta directamente como el baseline SASRec
en la Tabla IV, sin duplicar entrenamiento.

Por que estas 3 y no una reproduccion 1:1 de los papers originales:
se implementa una version SIMPLIFICADA pero fiel al mecanismo central
de cada arquitectura (asi quedo acordado), dado el tiempo disponible
para el Hito 3:

- LightGCN (He et al., 2020): version real y estandar -- SIN
  transformacion de features ni no-linealidad, solo propagacion de
  embeddings sobre la matriz de adyacencia normalizada simetrica
  (D^-1/2 A D^-1/2) y combinacion final como el promedio de todas las
  capas (incluida la capa 0). Es la version "ligera" que da nombre al
  modelo, tal cual el paper original.

- TransGNN (Zhang et al., 2024): combina una rama GNN con una rama
  Transformer que atiende sobre vecinos muestreados del grafo (en vez
  de una convolucion fija, el modelo APRENDE como ponderar cada vecino
  via autoatencion). Aqui se simplifica a: (1) las mismas capas de
  propagacion GCN que LightGCN para obtener embeddings con estructura
  de grafo, y (2) una capa de autoatencion (nn.TransformerEncoderLayer)
  sobre la secuencia [nodo, vecinos_muestreados] que refina cada
  embedding -- el mecanismo central del paper (atencion sobre vecinos
  estructurales) sin su muestreo jerarquico multi-salto completo.

- SIGformer (Chen et al., 2024): extiende la idea de un Transformer
  sobre el grafo agregando un embedding de SIGNO (positivo/negativo)
  sumado a cada token antes de la autoatencion, para que el modelo
  distinga explicitamente interacciones reales (signo +) de
  no-interacciones (signo -). Aqui se simplifica a: misma base GCN que
  LightGCN/TransGNN, mas una capa de autoatencion sobre la secuencia
  [nodo, vecinos_positivos (aristas reales), vecinos_negativos
  (no-aristas muestreadas)] con un embedding de signo aprendido --
  el aporte diferencial de SIGformer frente a TransGNN (atencion
  CON signo vs. SIN signo).

Las tres comparten la misma base GCN y el mismo protocolo de
entrenamiento (BPR, muestreo de negativos, particion leave-last-out),
por lo que la Tabla IV aisla el efecto de cada mecanismo de agregacion
sobre el mismo grafo y los mismos datos.

Entrada (igual que modelo_gnn.py):
    data/processed/{nombre}_edge_index.npy
    data/processed/{nombre}_id_maps.json
    data/processed/{nombre}_train.csv / _val.csv / _test.csv

Salida:
    data/processed/{nombre}_baselines.json  (Recall@10/NDCG@10 por modelo)
    Metricas impresas en consola

Uso (desde /workspace dentro del contenedor Docker):
    python src/modelo_baselines.py --dataset olist
    python src/modelo_baselines.py --dataset amazon
    python src/modelo_baselines.py --dataset all
    python src/modelo_baselines.py --dataset olist --modelos lightgcn
    python src/modelo_baselines.py --dataset all --epochs 20   (si 50 es muy lento)
-----------------------------------------------------------------------
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from modelo_gnn import SEED, PROC_DIR, DEVICE, set_seed, bpr_loss, muestrear_negativos
from metricas_eval import construir_catalogo, evaluar_scores

MODELOS_DISPONIBLES = ["lightgcn", "transgnn", "sigformer"]


# ------------------------------------------------------------------
# Datos y estructura del grafo (reutiliza el mismo formato que
# modelo_gnn.py, no se reescribe el pipeline de datos)
# ------------------------------------------------------------------

def cargar_datos(nombre, proc_dir=PROC_DIR):
    idmaps = json.load(open(f"{proc_dir}/{nombre}_id_maps.json"))
    edge_index = np.load(f"{proc_dir}/{nombre}_edge_index.npy")
    train = pd.read_csv(f"{proc_dir}/{nombre}_train.csv")
    test = pd.read_csv(f"{proc_dir}/{nombre}_test.csv")
    return idmaps, edge_index, train, test


def construir_adj_normalizada(edge_index_np, n_nodes, device):
    """D^-1/2 A D^-1/2 (normalizacion simetrica de LightGCN) como tensor
    disperso. edge_index ya viene no dirigido (ver preprocesar_dataset.py),
    asi que bincount(src) da el grado real de cada nodo."""
    src, dst = edge_index_np[0], edge_index_np[1]
    grados = np.bincount(src, minlength=n_nodes).astype(np.float64)
    grados[grados == 0] = 1.0  # nodos aislados (si los hubiera)
    inv_sqrt = 1.0 / np.sqrt(grados)
    valores = inv_sqrt[src] * inv_sqrt[dst]

    indices = torch.tensor(edge_index_np, dtype=torch.long)
    valores_t = torch.tensor(valores, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, valores_t, size=(n_nodes, n_nodes))
    return adj.coalesce().to(device)


def construir_listas_adyacencia(edge_index_np, n_nodes):
    """Lista de vecinos por nodo (vectorizado via ordenamiento, evita un
    loop Python por arista)."""
    src, dst = edge_index_np[0], edge_index_np[1]
    orden = np.argsort(src, kind="stable")
    src_ord, dst_ord = src[orden], dst[orden]
    limites = np.searchsorted(src_ord, np.arange(n_nodes + 1))
    return [dst_ord[limites[i]:limites[i + 1]] for i in range(n_nodes)]


def muestrear_vecinos(nodos_np, listas_adj, n_vecinos, rng):
    """Vecinos reales (aristas del grafo) de cada nodo, con reemplazo si
    hay menos de n_vecinos disponibles; si un nodo no tiene vecinos se
    autocompleta consigo mismo (equivalente a un self-loop)."""
    salida = np.empty((len(nodos_np), n_vecinos), dtype=np.int64)
    for i, n in enumerate(nodos_np):
        cand = listas_adj[n]
        if len(cand) == 0:
            salida[i] = n
        else:
            salida[i] = rng.choice(cand, size=n_vecinos, replace=len(cand) < n_vecinos)
    return salida


def muestrear_no_vecinos(nodos_np, listas_adj, n_vecinos, n_nodes, rng):
    """Nodos muestreados que NO son vecinos reales (arista negativa /
    sign=-1 para SIGformer)."""
    salida = np.empty((len(nodos_np), n_vecinos), dtype=np.int64)
    for i, n in enumerate(nodos_np):
        excluidos = set(listas_adj[n].tolist())
        excluidos.add(int(n))
        muestras = []
        while len(muestras) < n_vecinos:
            cand = int(rng.integers(0, n_nodes))
            if cand not in excluidos:
                muestras.append(cand)
        salida[i] = muestras
    return salida


# ------------------------------------------------------------------
# Modelos
# ------------------------------------------------------------------

class PropagacionGCN(nn.Module):
    """LightGCN: sin transformacion de features ni no-linealidad, solo
    propagacion lineal sobre la adyacencia normalizada y promedio de
    todas las capas (incluida la capa 0, el embedding original)."""

    def __init__(self, n_nodes, emb_dim=64, n_capas=3):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.n_capas = n_capas

    def forward(self, adj_norm):
        x = self.embedding.weight
        capas = [x]
        for _ in range(self.n_capas):
            x = torch.sparse.mm(adj_norm, x)
            capas.append(x)
        return torch.stack(capas, dim=0).mean(dim=0)


class RefinamientoEstructural(nn.Module):
    """TransGNN: autoatencion sobre [nodo, vecinos_muestreados] -- el
    mecanismo aprende a ponderar cada vecino en vez de usar pesos fijos
    de convolucion."""

    def __init__(self, emb_dim=64, n_heads=2):
        super().__init__()
        capa = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 2,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(capa, num_layers=1)

    def forward(self, emb_nodo, emb_vecinos):
        secuencia = torch.cat([emb_nodo.unsqueeze(1), emb_vecinos], dim=1)
        return self.encoder(secuencia)[:, 0]


class RefinamientoSignado(nn.Module):
    """SIGformer: igual que RefinamientoEstructural, pero cada token de
    la secuencia [nodo, vecinos_positivos, vecinos_negativos] recibe un
    embedding de signo (self / positivo / negativo) sumado antes de la
    autoatencion -- el aporte diferencial frente a TransGNN."""

    def __init__(self, emb_dim=64, n_heads=2):
        super().__init__()
        self.signo_emb = nn.Embedding(3, emb_dim)  # 0=self, 1=positivo, 2=negativo
        nn.init.normal_(self.signo_emb.weight, std=0.1)
        capa = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 2,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(capa, num_layers=1)

    def forward(self, emb_nodo, emb_pos, emb_neg):
        B = emb_nodo.shape[0]
        n_pos, n_neg = emb_pos.shape[1], emb_neg.shape[1]
        device = emb_nodo.device
        signos = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=device),
            torch.ones(B, n_pos, dtype=torch.long, device=device),
            torch.full((B, n_neg), 2, dtype=torch.long, device=device),
        ], dim=1)
        secuencia = torch.cat([emb_nodo.unsqueeze(1), emb_pos, emb_neg], dim=1)
        secuencia = secuencia + self.signo_emb(signos)
        return self.encoder(secuencia)[:, 0]


class TransGNN(nn.Module):
    def __init__(self, n_nodes, emb_dim=64, n_capas_gcn=2, n_vecinos=5, n_heads=2):
        super().__init__()
        self.gcn = PropagacionGCN(n_nodes, emb_dim, n_capas_gcn)
        self.refinamiento = RefinamientoEstructural(emb_dim, n_heads)
        self.n_vecinos = n_vecinos

    def forward(self, adj_norm, nodos, listas_adj, rng):
        emb_gcn = self.gcn(adj_norm)
        emb_nodo = emb_gcn[nodos]
        vecinos_np = muestrear_vecinos(nodos.cpu().numpy(), listas_adj, self.n_vecinos, rng)
        vecinos = torch.tensor(vecinos_np, dtype=torch.long, device=nodos.device)
        emb_vecinos = emb_gcn[vecinos]
        return self.refinamiento(emb_nodo, emb_vecinos)


class SIGformer(nn.Module):
    def __init__(self, n_nodes, emb_dim=64, n_capas_gcn=2, n_vecinos=5, n_heads=2):
        super().__init__()
        self.gcn = PropagacionGCN(n_nodes, emb_dim, n_capas_gcn)
        self.refinamiento = RefinamientoSignado(emb_dim, n_heads)
        self.n_vecinos = n_vecinos
        self.n_nodes = n_nodes

    def forward(self, adj_norm, nodos, listas_adj, rng):
        emb_gcn = self.gcn(adj_norm)
        emb_nodo = emb_gcn[nodos]
        nodos_np = nodos.cpu().numpy()
        pos_np = muestrear_vecinos(nodos_np, listas_adj, self.n_vecinos, rng)
        neg_np = muestrear_no_vecinos(nodos_np, listas_adj, self.n_vecinos, self.n_nodes, rng)
        pos = torch.tensor(pos_np, dtype=torch.long, device=nodos.device)
        neg = torch.tensor(neg_np, dtype=torch.long, device=nodos.device)
        emb_pos = emb_gcn[pos]
        emb_neg = emb_gcn[neg]
        return self.refinamiento(emb_nodo, emb_pos, emb_neg)


def construir_modelo(tipo, n_nodes, emb_dim, n_vecinos):
    if tipo == "lightgcn":
        return PropagacionGCN(n_nodes, emb_dim, n_capas=3)
    if tipo == "transgnn":
        return TransGNN(n_nodes, emb_dim, n_vecinos=n_vecinos)
    if tipo == "sigformer":
        return SIGformer(n_nodes, emb_dim, n_vecinos=n_vecinos)
    raise ValueError(f"modelo desconocido: {tipo}")


# ------------------------------------------------------------------
# Entrenamiento (mismo protocolo BPR que modelo_gnn.py)
# ------------------------------------------------------------------

def entrenar_baseline(nombre, tipo, epochs=50, emb_dim=64, lr=0.01, batch_size=1024,
                       n_vecinos=5, proc_dir=PROC_DIR):
    set_seed(SEED)
    idmaps, edge_index_np, train, test = cargar_datos(nombre, proc_dir)
    n_users, n_items = idmaps["n_users"], idmaps["n_items"]
    n_nodes = n_users + n_items

    adj_norm = construir_adj_normalizada(edge_index_np, n_nodes, DEVICE)
    listas_adj = construir_listas_adyacencia(edge_index_np, n_nodes)
    rng = np.random.default_rng(SEED)

    model = construir_modelo(tipo, n_nodes, emb_dim, n_vecinos).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    items_por_usuario = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    usuarios_train = train["user_idx"].to_numpy()
    items_train = train["item_idx"].to_numpy()
    n_train = len(train)

    print(f"[{nombre}][{tipo}] entrenando: {n_users:,} usuarios, {n_items:,} items, "
          f"{n_train:,} interacciones de train, {epochs} epocas")

    model.train()
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n_train)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            u_batch = usuarios_train[idx]
            pos_batch = items_train[idx]
            neg_batch = muestrear_negativos(u_batch, n_items, items_por_usuario)

            if tipo == "lightgcn":
                emb = model(adj_norm)
                u_emb = emb[u_batch]
                pos_emb = emb[pos_batch + n_users]
                neg_emb = emb[neg_batch + n_users]
            else:
                nodos_np = np.concatenate([u_batch, pos_batch + n_users, neg_batch + n_users])
                nodos = torch.tensor(nodos_np, dtype=torch.long, device=DEVICE)
                salida = model(adj_norm, nodos, listas_adj, rng)
                B = len(u_batch)
                u_emb, pos_emb, neg_emb = salida[:B], salida[B:2 * B], salida[2 * B:]

            loss = bpr_loss(u_emb, pos_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == epochs:
            print(f"[{nombre}][{tipo}] epoca {epoch}/{epochs} - loss BPR promedio: "
                  f"{total_loss / n_batches:.4f}")

    return model, adj_norm, listas_adj, rng, idmaps, train, test


# ------------------------------------------------------------------
# Evaluacion: Recall@K y NDCG@K (ranking completo, igual protocolo que
# modelo_gnn.py)
# ------------------------------------------------------------------

@torch.no_grad()
def evaluar_baseline(model, tipo, adj_norm, listas_adj, rng, idmaps, train, test, k=10):
    model.eval()
    n_users, n_items = idmaps["n_users"], idmaps["n_items"]
    n_nodes = n_users + n_items

    if tipo == "lightgcn":
        node_emb = model(adj_norm)
    else:
        todos_nodos = torch.arange(n_nodes, dtype=torch.long, device=DEVICE)
        node_emb = model(adj_norm, todos_nodos, listas_adj, rng)

    user_emb = node_emb[:n_users]
    item_emb = node_emb[n_users:]

    catalogo = construir_catalogo(train)
    usuarios_a_evaluar = test["user_idx"].unique()
    scores_por_usuario = {int(u): (user_emb[int(u)] @ item_emb.T).cpu().numpy()
                           for u in usuarios_a_evaluar}

    return evaluar_scores(scores_por_usuario, train, test, catalogo, k=k)


# ------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------

def procesar(nombre, modelos=MODELOS_DISPONIBLES, epochs=50, k=10, emb_dim=64,
             n_vecinos=5, proc_dir=PROC_DIR):
    resultados = {}
    for tipo in modelos:
        model, adj_norm, listas_adj, rng, idmaps, train, test = entrenar_baseline(
            nombre, tipo, epochs=epochs, emb_dim=emb_dim, n_vecinos=n_vecinos,
            proc_dir=proc_dir)
        metricas = evaluar_baseline(model, tipo, adj_norm, listas_adj, rng, idmaps,
                                     train, test, k=k)
        print(f"[{nombre}][{tipo}] resultados en test -> {metricas}\n")
        resultados[tipo] = metricas

    with open(f"{proc_dir}/{nombre}_baselines.json", "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"[{nombre}] resumen guardado en {proc_dir}/{nombre}_baselines.json\n")
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"], default="all")
    parser.add_argument("--modelos", nargs="+", choices=MODELOS_DISPONIBLES,
                         default=MODELOS_DISPONIBLES,
                         help="por defecto corre los 3 (lightgcn transgnn sigformer)")
    parser.add_argument("--epochs", type=int, default=50,
                         help="50 para ser comparable con las Tablas I/II/III; "
                              "si transgnn/sigformer tardan demasiado en tu Docker, "
                              "baja esto a 20 (el mismo ajuste que se hizo antes)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_vecinos", type=int, default=5,
                         help="tamano de la ventana de vecinos muestreados para "
                              "transgnn/sigformer")
    args = parser.parse_args()

    if args.dataset in ("olist", "all"):
        procesar("olist", modelos=args.modelos, epochs=args.epochs, k=args.k,
                  n_vecinos=args.n_vecinos)
    if args.dataset in ("amazon", "all"):
        procesar("amazon", modelos=args.modelos, epochs=args.epochs, k=args.k,
                  n_vecinos=args.n_vecinos)
