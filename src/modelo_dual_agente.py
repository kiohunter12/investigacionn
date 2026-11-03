"""
modelo_dual_agente.py
-----------------------------------------------------------------------
H4 del PPI: ¿la arquitectura dual-agente (Selector + Recomendador)
supera al componente unico (monolitico)? Tabla V del short paper.

Las dos configuraciones comparten EXACTAMENTE el mismo backbone (rama
GNN congelada, compuerta de fusion CompuertaFusion/ModeloHibridoGate de
modelo_hibrido_gate.py, mismo seed=42, mismas epocas) y difieren en un
unico punto, para aislar el efecto de la decision arquitectonica:

- DUAL-AGENTE: el Transformer usa el Selector (Linear-ReLU-Linear-
  Sigmoid) que pondera cada interaccion pasada ANTES de que entre al
  encoder -- dos componentes especializados (Selector = agente que
  decide que interacciones importan; Transformer+GNN+gate = agente
  que arma la recomendacion), tal como ya viene siendo reportado en
  las Tablas I/III/IV/VI (reutiliza trans_mod.SecuenciaTransformer
  sin cambios).
- MONOLITICO: el Transformer recibe la secuencia cruda, SIN ninguna
  etapa de ponderacion previa -- un unico componente hace todo el
  trabajo de principio a fin (se define aqui
  SecuenciaTransformerMonolitico, una copia de SecuenciaTransformer
  con el Selector removido; no se modifica modelo_transformer.py).

Ambas configuraciones se evaluan por usuario (no solo el promedio) para
poder aplicar una prueba estadistica PAREADA:
    - McNemar (aciertos/fallos de Recall@10, datos binarios pareados)
    - Wilcoxon signed-rank (NDCG@10, continuo pareado)
sobre el MISMO conjunto de usuarios de test, bajo H0: "no hay diferencia
sistematica entre las dos configuraciones".

Entrada: la misma que modelo_hibrido_gate.py (data/processed/{nombre}_*).
Salida:
    data/processed/{nombre}_dual_agente.json
    Metricas + resultado de las pruebas estadisticas impresos en consola

Uso (desde /workspace dentro del contenedor Docker):
    python src/modelo_dual_agente.py --dataset olist
    python src/modelo_dual_agente.py --dataset amazon
    python src/modelo_dual_agente.py --dataset all
-----------------------------------------------------------------------
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon, binomtest

import modelo_gnn as gnn_mod
import modelo_transformer as trans_mod
from modelo_hibrido_gate import ModeloHibridoGate, bpr_loss, SEED, PROC_DIR, DEVICE, set_seed

CONFIGURACIONES = ["dual_agente", "monolitico"]


# ------------------------------------------------------------------
# Variante MONOLITICA del Transformer: identica a
# trans_mod.SecuenciaTransformer salvo que NO tiene el Selector -- la
# secuencia entra cruda al encoder. Es la unica diferencia
# arquitectonica frente a la version dual-agente.
# ------------------------------------------------------------------

class SecuenciaTransformerMonolitico(nn.Module):
    def __init__(self, n_items, max_len=None, emb_dim=64, n_heads=2,
                 n_layers=2, dropout=0.2):
        super().__init__()
        max_len = max_len or trans_mod.MAX_LEN
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        capa = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(capa, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        batch, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(batch, L)
        x = self.item_emb(seq) + self.pos_emb(pos)
        # SIN selector: a diferencia de SecuenciaTransformer, aqui no hay
        # ninguna etapa previa de ponderacion de relevancia -- esta es la
        # unica diferencia frente a la version dual-agente.
        x = self.dropout(x)
        pad_mask = (seq == 0)
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        return h


def entrenar_transformer_monolitico(nombre, epochs=20, emb_dim=64, lr=0.001,
                                     batch_size=128, proc_dir=PROC_DIR):
    """Mismo loop de entrenamiento que trans_mod.entrenar(), instanciando
    SecuenciaTransformerMonolitico en vez de SecuenciaTransformer."""
    set_seed(SEED)
    idmaps = json.load(open(f"{proc_dir}/{nombre}_id_maps.json"))
    n_items = idmaps["n_items"]

    train = pd.read_csv(f"{proc_dir}/{nombre}_train.csv")
    test = pd.read_csv(f"{proc_dir}/{nombre}_test.csv")

    secuencias = trans_mod.cargar_secuencias_train(train)
    secuencias = {u: s for u, s in secuencias.items() if len(s) >= 2}
    usuarios = list(secuencias.keys())

    model = SecuenciaTransformerMonolitico(
        n_items, max_len=trans_mod.MAX_LEN, emb_dim=emb_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[{nombre}] entrenando Transformer MONOLITICO (sin selector): "
          f"{len(usuarios):,} usuarios, {n_items:,} items, {epochs} epocas")

    model.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(usuarios)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(usuarios), batch_size):
            batch_users = usuarios[start:start + batch_size]
            entradas, objetivos = [], []
            for u in batch_users:
                seq = secuencias[u]
                entradas.append(trans_mod.secuencia_a_tensor(seq[:-1])[0])
                objetivos.append(trans_mod.secuencia_a_tensor(seq[1:])[0])

            entradas_t = torch.tensor(entradas, dtype=torch.long, device=DEVICE)
            objetivos_t = torch.tensor(objetivos, dtype=torch.long, device=DEVICE)
            negativos = np.random.randint(1, n_items + 1, size=objetivos_t.shape)
            negativos_t = torch.tensor(negativos, dtype=torch.long, device=DEVICE)

            h = model(entradas_t)
            mask_validos = objetivos_t != 0
            pos_emb = model.item_emb(objetivos_t)
            neg_emb = model.item_emb(negativos_t)

            h_v = h[mask_validos]
            pos_v = pos_emb[mask_validos]
            neg_v = neg_emb[mask_validos]
            if h_v.shape[0] == 0:
                continue

            loss = trans_mod.bpr_loss(h_v, pos_v, neg_v)
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
# Entrenamiento de la fusion con gate, identico a
# modelo_hibrido_gate.entrenar_fusion() pero recibiendo el trans_model
# ya entrenado (sea dual-agente o monolitico) en vez de entrenarlo
# adentro -- asi no se duplica logica para cada variante.
# ------------------------------------------------------------------

def entrenar_fusion_con_rama(nombre, gnn_node_emb, trans_model, idmaps, train,
                              secuencias, epochs_fusion=15, lr=0.001,
                              batch_size=128, freeze_ramas=True):
    n_users, n_items = idmaps["n_users"], idmaps["n_items"]

    modelo = ModeloHibridoGate(gnn_node_emb, trans_model, n_users,
                                emb_dim=gnn_node_emb.shape[1],
                                freeze_ramas=freeze_ramas).to(DEVICE)

    parametros_entrenables = [p for p in modelo.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parametros_entrenables, lr=lr)

    secuencias_train = {u: s for u, s in secuencias.items() if len(s) >= 2}
    usuarios = list(secuencias_train.keys())

    modelo.train()
    if freeze_ramas:
        modelo.trans_model.eval()

    for epoch in range(1, epochs_fusion + 1):
        np.random.shuffle(usuarios)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(usuarios), batch_size):
            batch_users = usuarios[start:start + batch_size]
            entradas, objetivos = [], []
            for u in batch_users:
                seq = secuencias_train[u]
                entradas.append(trans_mod.secuencia_a_tensor(seq[:-1])[0])
                objetivos.append(trans_mod.secuencia_a_tensor(seq[1:])[0])

            entradas_t = torch.tensor(entradas, dtype=torch.long, device=DEVICE)
            objetivos_t = torch.tensor(objetivos, dtype=torch.long, device=DEVICE)
            u_idx_t = torch.tensor(batch_users, dtype=torch.long, device=DEVICE)

            negativos = np.random.randint(0, n_items, size=objetivos_t.shape)
            negativos_t = torch.tensor(negativos, dtype=torch.long, device=DEVICE)

            h = modelo.estado_transformer(entradas_t)
            mask_validos = objetivos_t != 0
            u_idx_expandido = u_idx_t.unsqueeze(1).expand(-1, h.shape[1])

            h_v = h[mask_validos]
            u_idx_v = u_idx_expandido[mask_validos]
            pos_v = (objetivos_t[mask_validos] - 1)
            neg_v = negativos_t[mask_validos]
            if h_v.shape[0] == 0:
                continue

            fused_u = modelo.fusionar_usuario(u_idx_v, h_v)
            fused_pos = modelo.fusionar_item(pos_v)
            fused_neg = modelo.fusionar_item(neg_v)

            loss = bpr_loss(fused_u, fused_pos, fused_neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == epochs_fusion:
            print(f"[{nombre}] fusion(gate) epoca {epoch}/{epochs_fusion} - "
                  f"loss BPR promedio: {total_loss / max(1, n_batches):.4f}")

    return modelo


def entrenar_configuracion(nombre, tipo, epochs_ramas=50, epochs_fusion=15,
                            proc_dir=PROC_DIR):
    assert tipo in CONFIGURACIONES
    set_seed(SEED)

    print(f"[{nombre}][{tipo}] (1/2) entrenando rama GNN...")
    gnn_model, gnn_edge_index, gnn_idmaps, gnn_train, gnn_val, gnn_test = \
        gnn_mod.entrenar(nombre, epochs=epochs_ramas)

    if tipo == "dual_agente":
        print(f"[{nombre}][{tipo}] (2/2) entrenando Transformer CON selector...")
        trans_model, idmaps, train, test, secuencias = \
            trans_mod.entrenar(nombre, epochs=epochs_ramas, proc_dir=proc_dir)
    else:
        print(f"[{nombre}][{tipo}] (2/2) entrenando Transformer SIN selector...")
        trans_model, idmaps, train, test, secuencias = \
            entrenar_transformer_monolitico(nombre, epochs=epochs_ramas, proc_dir=proc_dir)

    with torch.no_grad():
        gnn_model.eval()
        gnn_node_emb = gnn_model(gnn_edge_index).detach()

    modelo = entrenar_fusion_con_rama(
        nombre, gnn_node_emb, trans_model, idmaps, train, secuencias,
        epochs_fusion=epochs_fusion)

    return modelo, idmaps, train, test, secuencias


# ------------------------------------------------------------------
# Evaluacion POR USUARIO (no solo el promedio), para poder aplicar las
# pruebas estadisticas pareadas.
# ------------------------------------------------------------------

@torch.no_grad()
def evaluar_por_usuario(modelo, idmaps, train, test, k=10):
    modelo.eval()
    n_items = idmaps["n_items"]

    secuencias_train = trans_mod.cargar_secuencias_train(train)
    items_test = test.set_index("user_idx")["item_idx"].to_dict()

    todos_items = torch.arange(n_items, dtype=torch.long, device=DEVICE)
    fused_item_all = modelo.fusionar_item(todos_items)

    resultados = {}  # user_idx -> (hit: 0/1, ndcg: float)
    for u, item_objetivo in items_test.items():
        seq = secuencias_train.get(u)
        if not seq:
            continue
        seq_tensor, longitud_real = trans_mod.secuencia_a_tensor(seq)
        entrada = torch.tensor([seq_tensor], dtype=torch.long, device=DEVICE)
        h = modelo.estado_transformer(entrada)[0, longitud_real - 1]

        u_idx_t = torch.tensor([u], dtype=torch.long, device=DEVICE)
        fused_u = modelo.fusionar_usuario(u_idx_t, h.unsqueeze(0))[0]

        scores = (fused_item_all @ fused_u).cpu().numpy()
        vistos = set(seq)
        if vistos:
            scores[list(vistos)] = -np.inf

        top_k = np.argpartition(-scores, k)[:k]
        top_k = top_k[np.argsort(-scores[top_k])]

        acierto = item_objetivo in top_k
        if acierto:
            rank = int(np.where(top_k == item_objetivo)[0][0])
            ndcg = 1.0 / np.log2(rank + 2)
        else:
            ndcg = 0.0
        resultados[u] = (1.0 if acierto else 0.0, ndcg)

    return resultados


def resumir(resultados):
    hits = [h for h, _ in resultados.values()]
    ndcgs = [n for _, n in resultados.values()]
    return {
        "usuarios_evaluados": len(resultados),
        "recall@10": float(np.mean(hits)) if hits else 0.0,
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


# ------------------------------------------------------------------
# Prueba estadistica pareada: McNemar (aciertos binarios) + Wilcoxon
# signed-rank (NDCG continuo), ambas sobre el conjunto de usuarios
# evaluados por AMBAS configuraciones.
# ------------------------------------------------------------------

def mcnemar_exacto(hits_a, hits_b):
    """McNemar exacto sobre pares discordantes (a favor de A vs a favor
    de B), via test binomial exacto (igual que statsmodels con
    exact=True, sin depender de esa libreria)."""
    n01 = sum(1 for a, b in zip(hits_a, hits_b) if a == 0 and b == 1)  # gana B
    n10 = sum(1 for a, b in zip(hits_a, hits_b) if a == 1 and b == 0)  # gana A
    n_discordantes = n01 + n10
    if n_discordantes == 0:
        return {"n01": n01, "n10": n10, "p_valor": 1.0}
    p = binomtest(min(n01, n10), n_discordantes, 0.5, alternative="two-sided").pvalue
    return {"n01_gana_monolitico": n01, "n10_gana_dual_agente": n10, "p_valor": float(p)}


def comparar(nombre, res_dual, res_mono):
    usuarios_comunes = sorted(set(res_dual.keys()) & set(res_mono.keys()))
    hits_dual = [res_dual[u][0] for u in usuarios_comunes]
    hits_mono = [res_mono[u][0] for u in usuarios_comunes]
    ndcg_dual = [res_dual[u][1] for u in usuarios_comunes]
    ndcg_mono = [res_mono[u][1] for u in usuarios_comunes]

    mcnemar = mcnemar_exacto(hits_mono, hits_dual)  # a=monolitico, b=dual_agente

    diffs = np.array(ndcg_dual) - np.array(ndcg_mono)
    if np.all(diffs == 0):
        wilcoxon_res = {"estadistico": 0.0, "p_valor": 1.0}
    else:
        stat, p = wilcoxon(ndcg_dual, ndcg_mono, zero_method="wilcox")
        wilcoxon_res = {"estadistico": float(stat), "p_valor": float(p)}

    return {
        "n_usuarios_comparados": len(usuarios_comunes),
        "mcnemar_recall@10": mcnemar,
        "wilcoxon_ndcg@10": wilcoxon_res,
    }


# ------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------

def procesar(nombre, epochs_ramas=50, epochs_fusion=15, k=10, proc_dir=PROC_DIR):
    modelo_dual, idmaps, train, test, secuencias = entrenar_configuracion(
        nombre, "dual_agente", epochs_ramas=epochs_ramas,
        epochs_fusion=epochs_fusion, proc_dir=proc_dir)
    res_dual = evaluar_por_usuario(modelo_dual, idmaps, train, test, k=k)
    metricas_dual = resumir(res_dual)
    print(f"[{nombre}][dual_agente] resultados en test -> {metricas_dual}\n")

    modelo_mono, idmaps2, train2, test2, secuencias2 = entrenar_configuracion(
        nombre, "monolitico", epochs_ramas=epochs_ramas,
        epochs_fusion=epochs_fusion, proc_dir=proc_dir)
    res_mono = evaluar_por_usuario(modelo_mono, idmaps2, train2, test2, k=k)
    metricas_mono = resumir(res_mono)
    print(f"[{nombre}][monolitico] resultados en test -> {metricas_mono}\n")

    prueba = comparar(nombre, res_dual, res_mono)
    print(f"[{nombre}] prueba estadistica pareada (H4) -> {prueba}\n")

    resumen = {
        "dataset": nombre,
        "dual_agente": metricas_dual,
        "monolitico": metricas_mono,
        "prueba_estadistica": prueba,
    }
    with open(f"{proc_dir}/{nombre}_dual_agente.json", "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"[{nombre}] resumen guardado en {proc_dir}/{nombre}_dual_agente.json\n")
    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"], default="all")
    parser.add_argument("--epochs_ramas", type=int, default=50,
                         help="50 para ser comparable con el resto de tablas del paper")
    parser.add_argument("--epochs_fusion", type=int, default=15)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if args.dataset in ("olist", "all"):
        procesar("olist", epochs_ramas=args.epochs_ramas, epochs_fusion=args.epochs_fusion, k=args.k)
    if args.dataset in ("amazon", "all"):
        procesar("amazon", epochs_ramas=args.epochs_ramas, epochs_fusion=args.epochs_fusion, k=args.k)
