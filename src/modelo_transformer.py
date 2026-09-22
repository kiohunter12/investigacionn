"""
modelo_transformer.py
-----------------------------------------------------------------------
Rama Transformer del modelo hibrido (Semanas 3-4 del cronograma).

Encoder tipo Transformer autoregresivo (estilo SASRec) + SELECTOR de
relevancia, adaptando el "state tracker" de DARLR (idea base del PPI):
en vez de mirar el grafo de conexiones (rama GNN), esta rama mira el
ORDEN en que cada usuario fue interactuando con productos, y aprende a
predecir cual es el siguiente producto probable.

El SELECTOR es una mini-red (Linear -> ReLU -> Linear -> Sigmoid) que,
para cada interaccion pasada de la secuencia, calcula un peso de
relevancia entre 0 y 1 (cuanto le importa esa compra especifica para
predecir la siguiente) ANTES de que esa informacion entre al Transformer.
Se entrena junto con todo lo demas (no necesita supervision aparte) y,
ademas de mejorar la senal de entrada, sus pesos son interpretables:
sirven como insumo para el modulo de explicabilidad (XAI) del proyecto,
mostrando que interacciones pasadas pesaron mas en cada recomendacion.

Entrada (generada por el pipeline de datos, Hito 1):
    data/processed/{nombre}_train.csv / _test.csv
    data/processed/{nombre}_id_maps.json

Salida:
    data/processed/{nombre}_transformer_embeddings.npy  (embeddings de item)
    Metricas Recall@K / NDCG@K impresas en consola (mismo protocolo que
    modelo_gnn.py, para que ambas ramas sean comparables)

Uso (desde /workspace dentro del contenedor Docker):
    python src/modelo_transformer.py --dataset olist
    python src/modelo_transformer.py --dataset amazon
    python src/modelo_transformer.py --dataset all
-----------------------------------------------------------------------
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
PROC_DIR = "data/processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 50  # longitud maxima de secuencia que ve el modelo (recorta/rellena)


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------------------
# Selector de relevancia (idea base: state tracker de DARLR)
# ------------------------------------------------------------------

class Selector(nn.Module):
    """Para cada posicion de la secuencia, aprende un peso de relevancia
    en (0,1) que pondera esa interaccion antes de pasarla al Transformer.
    No necesita supervision aparte: se entrena junto con el resto del
    modelo (la senal viene del mismo objetivo de prediccion). Los pesos
    aprendidos son interpretables (insumo para XAI)."""

    def __init__(self, emb_dim):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq, emb_dim)
        peso = self.red(x)  # (batch, seq, 1) -- relevancia de cada interaccion
        return x * peso, peso.squeeze(-1)


# ------------------------------------------------------------------
# Modelo: Selector + Transformer autoregresivo estilo SASRec
# ------------------------------------------------------------------

class SecuenciaTransformer(nn.Module):
    """item_idx van de 0..n_items-1; se les suma 1 porque 0 se reserva
    para relleno (padding) de secuencias cortas."""

    def __init__(self, n_items, max_len=MAX_LEN, emb_dim=64, n_heads=2,
                 n_layers=2, dropout=0.2):
        super().__init__()
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.selector = Selector(emb_dim)
        capa = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(capa, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, devolver_pesos_selector=False):
        # seq: (batch, max_len) con 0 = padding, items desplazados +1
        batch, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(batch, L)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x, pesos_selector = self.selector(x)  # pondera cada interaccion por relevancia
        x = self.dropout(x)

        pad_mask = (seq == 0)  # True donde hay padding
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)

        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        if devolver_pesos_selector:
            return h, pesos_selector
        return h  # (batch, max_len, emb_dim) -- estado oculto en cada posicion


def bpr_loss(h_pos, item_emb_pos, item_emb_neg):
    pos_scores = (h_pos * item_emb_pos).sum(-1)
    neg_scores = (h_pos * item_emb_neg).sum(-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


# ------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------

def cargar_secuencias_train(train_df):
    """Reconstruye, a partir de train.csv (ya excluye val/test por
    leave-last-out), la secuencia cronologica de items por usuario."""
    df_sorted = train_df.sort_values(["user_idx", "timestamp"])
    return df_sorted.groupby("user_idx")["item_idx"].apply(list).to_dict()


def secuencia_a_tensor(seq, max_len=MAX_LEN):
    """Recorta a los ultimos max_len items y rellena con 0 A LA DERECHA
    (no a la izquierda). Los item_idx reales se desplazan +1 (0 queda
    libre para padding).

    IMPORTANTE: con mascara causal, rellenar a la IZQUIERDA hace que las
    posiciones de relleno iniciales no tengan ninguna clave valida a la
    que atender (todas sus posiciones pasadas tambien son relleno), lo
    que produce NaN en la atencion (softmax de una fila completamente
    enmascarada) y ese NaN contamina el resto de la secuencia. Rellenando
    a la derecha, toda posicion (real o de relleno) siempre tiene al
    menos el primer token real como clave valida hacia el pasado."""
    seq = seq[-max_len:]
    seq = [i + 1 for i in seq]
    pad = [0] * (max_len - len(seq))
    return seq + pad, len(seq)  # tambien se devuelve la longitud real


# ------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------

def entrenar(nombre, epochs=20, emb_dim=64, lr=0.001, batch_size=128,
             proc_dir=PROC_DIR):
    set_seed(SEED)
    idmaps = json.load(open(f"{proc_dir}/{nombre}_id_maps.json"))
    n_items = idmaps["n_items"]

    train = pd.read_csv(f"{proc_dir}/{nombre}_train.csv")
    test = pd.read_csv(f"{proc_dir}/{nombre}_test.csv")

    secuencias = cargar_secuencias_train(train)
    # solo usuarios con al menos 2 interacciones en train (se necesita
    # 1 de entrada + 1 objetivo para poder entrenar con esa secuencia)
    secuencias = {u: s for u, s in secuencias.items() if len(s) >= 2}
    usuarios = list(secuencias.keys())

    model = SecuenciaTransformer(n_items, max_len=MAX_LEN, emb_dim=emb_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[{nombre}] entrenando Transformer: {len(usuarios):,} usuarios con "
          f"secuencia util, {n_items:,} items, {epochs} epocas")

    model.train()
    n_batches_total = max(1, len(usuarios) // batch_size)
    for epoch in range(1, epochs + 1):
        np.random.shuffle(usuarios)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(usuarios), batch_size):
            batch_users = usuarios[start:start + batch_size]
            entradas, objetivos, vistos_batch = [], [], []
            for u in batch_users:
                seq = secuencias[u]
                entradas.append(secuencia_a_tensor(seq[:-1])[0])
                objetivos.append(secuencia_a_tensor(seq[1:])[0])
                vistos_batch.append(set(seq))

            entradas = torch.tensor(entradas, dtype=torch.long, device=DEVICE)
            objetivos_t = torch.tensor(objetivos, dtype=torch.long, device=DEVICE)

            # negativos: un item aleatorio no visto por el usuario, por posicion
            negativos = np.random.randint(1, n_items + 1, size=objetivos_t.shape)
            negativos_t = torch.tensor(negativos, dtype=torch.long, device=DEVICE)

            h = model(entradas)  # (batch, max_len, emb_dim)
            mask_validos = objetivos_t != 0  # ignorar posiciones de padding

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

        if epoch % 5 == 0 or epoch == epochs:
            print(f"[{nombre}] epoca {epoch}/{epochs} - loss BPR promedio: "
                  f"{total_loss / max(1, n_batches):.4f}")

    return model, idmaps, train, test, secuencias


# ------------------------------------------------------------------
# Evaluacion: Recall@K y NDCG@K (mismo protocolo que modelo_gnn.py)
# ------------------------------------------------------------------

@torch.no_grad()
def evaluar(model, idmaps, train, test, k=10):
    model.eval()
    n_items = idmaps["n_items"]

    secuencias_train = cargar_secuencias_train(train)
    items_test = test.set_index("user_idx")["item_idx"].to_dict()

    item_emb_all = model.item_emb.weight[1:]  # sin la fila de padding (0)

    recalls, ndcgs = [], []
    for u, item_objetivo in items_test.items():
        seq = secuencias_train.get(u)
        if not seq:
            continue
        seq_tensor, longitud_real = secuencia_a_tensor(seq)
        entrada = torch.tensor([seq_tensor], dtype=torch.long, device=DEVICE)
        # con relleno a la derecha, la ultima posicion REAL (no de
        # padding) queda en el indice (longitud_real - 1), no en -1
        h = model(entrada)[0, longitud_real - 1]

        scores = (item_emb_all @ h).cpu().numpy()
        vistos = set(seq)
        if vistos:
            scores[list(vistos)] = -np.inf

        top_k = np.argpartition(-scores, k)[:k]
        top_k = top_k[np.argsort(-scores[top_k])]

        acierto = item_objetivo in top_k
        recalls.append(1.0 if acierto else 0.0)
        if acierto:
            rank = int(np.where(top_k == item_objetivo)[0][0])
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            ndcgs.append(0.0)

    return {
        "usuarios_evaluados": len(recalls),
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


@torch.no_grad()
def ejemplo_explicabilidad(model, train, n_items_mostrar=8):
    """Imprime, para un usuario de ejemplo, el peso de relevancia que el
    selector le dio a cada una de sus ultimas compras (insumo XAI: que
    interacciones pasadas pesaron mas para la prediccion)."""
    model.eval()
    secuencias_train = cargar_secuencias_train(train)
    u = next((u for u, s in secuencias_train.items() if len(s) >= 3), None)
    if u is None:
        return
    seq = secuencias_train[u]
    seq_tensor, longitud_real = secuencia_a_tensor(seq)
    entrada = torch.tensor([seq_tensor], dtype=torch.long, device=DEVICE)
    _, pesos = model(entrada, devolver_pesos_selector=True)
    pesos = pesos[0, :longitud_real].cpu().numpy()  # solo posiciones reales

    ultimos = list(zip(seq[-n_items_mostrar:], pesos[-n_items_mostrar:]))
    print(f"  ejemplo de explicabilidad (usuario {u}, selector): "
          + ", ".join(f"item {it}->{p:.2f}" for it, p in ultimos))


# ------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------

def procesar(nombre, epochs=20, k=10, proc_dir=PROC_DIR):
    model, idmaps, train, test, secuencias = entrenar(
        nombre, epochs=epochs, proc_dir=proc_dir)

    metricas = evaluar(model, idmaps, train, test, k=k)
    print(f"[{nombre}] resultados en test -> {metricas}")
    print(f"[{nombre}]", end=" ")
    ejemplo_explicabilidad(model, train)

    with torch.no_grad():
        embeddings = model.item_emb.weight[1:].cpu().numpy()  # 1 vector por item
    np.save(f"{proc_dir}/{nombre}_transformer_embeddings.npy", embeddings)
    print(f"[{nombre}] embeddings guardados en "
          f"{proc_dir}/{nombre}_transformer_embeddings.npy "
          f"(shape={embeddings.shape})\n")
    return metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if args.dataset in ("olist", "all"):
        procesar("olist", epochs=args.epochs, k=args.k)
    if args.dataset in ("amazon", "all"):
        procesar("amazon", epochs=args.epochs, k=args.k)
