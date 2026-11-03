"""
modelo_hibrido_gate.py
-----------------------------------------------------------------------
Capa de fusion hibrida GNN + Transformer CON COMPUERTA APRENDIDA
(ablation sobre modelo_hibrido.py, Semanas 3-4 del cronograma).

modelo_hibrido.py fusiona por concatenacion + proyeccion lineal FIJA.
Los resultados mostraron que esa fusion queda por debajo del GNN solo
en los dos datasets (olist y amazon): la proyeccion fija no tiene forma
de "bajarle el volumen" a la rama Transformer cuando esta es mas debil
(su loss BPR final es ~2 ordenes de magnitud peor que el del GNN en
ambos datasets). Este script reemplaza la proyeccion fija por una
COMPUERTA (gate) aprendida que pondera dinamicamente cada rama:

    peso = sigmoid(MLP(concat(emb_gnn, emb_transformer)))   -> escalar en (0,1)
    fusionado = peso * emb_gnn + (1 - peso) * emb_transformer

Un gate para el lado usuario y otro para el lado item (son problemas
distintos: el gate de usuario pondera "estado GNN estatico" vs "estado
Transformer dinamico"; el de item pondera dos embeddings estaticos).

Por que esto es mejor que la proyeccion fija:
- Si el Transformer no aporta señal para un usuario/item dado, el gate
  puede aprender peso ~= 1 (confiar solo en GNN) en vez de forzar una
  mezcla ciega de 128 dims -> 64 dims como hacia modelo_hibrido.py.
- El peso aprendido es ademas un numero INTERPRETABLE por si mismo:
  "en promedio el modelo confia X% en el GNN y (1-X)% en el Transformer"
  es un insumo directo para la seccion de explicabilidad (XAI) del PPI,
  complementario a los pesos del Selector de modelo_transformer.py.

Igual que modelo_hibrido.py: reutiliza las ramas YA VALIDADAS via sus
funciones entrenar() (no se reescribe modelo_gnn.py ni
modelo_transformer.py), y por default las congela (--fine_tune_ramas
para descongelar).

Entrada: la misma que las ramas (data/processed/{nombre}_*).
Salida:
    data/processed/{nombre}_hibrido_gate_embeddings_usuario.npy
    data/processed/{nombre}_hibrido_gate_embeddings_item.npy
    Metricas Recall@K / NDCG@K + peso promedio del gate (usuario/item)
    impresos en consola, mismo protocolo que las otras tres variantes
    (GNN solo, Transformer solo, Hibrido concat+proyeccion) -> las
    cuatro son directamente comparables.

Uso (desde /workspace dentro del contenedor Docker):
    python src/modelo_hibrido_gate.py --dataset olist
    python src/modelo_hibrido_gate.py --dataset amazon
    python src/modelo_hibrido_gate.py --dataset all
    python src/modelo_hibrido_gate.py --dataset olist --fine_tune_ramas
-----------------------------------------------------------------------
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import modelo_gnn as gnn_mod
import modelo_transformer as trans_mod
from metricas_eval import construir_catalogo, evaluar_scores

SEED = 42
PROC_DIR = "data/processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------------------
# Compuerta de fusion: pondera dinamicamente dos embeddings del mismo
# tamaño (uno por rama) en vez de concatenar+proyectar a ciegas.
# ------------------------------------------------------------------

class CompuertaFusion(nn.Module):
    """peso = sigmoid(MLP(concat(emb_a, emb_b))) en (0,1).
    fusionado = peso * emb_a + (1 - peso) * emb_b.
    Devuelve tambien 'peso' (por fila) para poder reportarlo como
    insumo de explicabilidad."""

    def __init__(self, emb_dim):
        super().__init__()
        self.red_gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, emb_a, emb_b):
        peso = self.red_gate(torch.cat([emb_a, emb_b], dim=-1))  # (N, 1)
        fusionado = peso * emb_a + (1.0 - peso) * emb_b
        return fusionado, peso.squeeze(-1)


class ModeloHibridoGate(nn.Module):
    """Igual que ModeloHibrido (modelo_hibrido.py) pero con compuerta
    en vez de concat+proyeccion lineal fija. gnn_node_emb y trans_model
    se usan exactamente igual que alli."""

    def __init__(self, gnn_node_emb, trans_model, n_users, emb_dim=64,
                 freeze_ramas=True):
        super().__init__()
        self.register_buffer("gnn_node_emb", gnn_node_emb)
        self.trans_model = trans_model
        self.n_users = n_users

        if freeze_ramas:
            for p in self.trans_model.parameters():
                p.requires_grad = False

        self.gate_usuario = CompuertaFusion(emb_dim)
        self.gate_item = CompuertaFusion(emb_dim)

    def fusionar_usuario(self, u_idx, h_transformer, devolver_peso=False):
        gnn_u = self.gnn_node_emb[u_idx]  # (N, emb_dim) -- estatico
        fusionado, peso = self.gate_usuario(gnn_u, h_transformer)
        if devolver_peso:
            return fusionado, peso
        return fusionado

    def fusionar_item(self, item_idx, devolver_peso=False):
        gnn_i = self.gnn_node_emb[self.n_users + item_idx]
        trans_i = self.trans_model.item_emb(item_idx + 1)  # +1: offset de padding
        fusionado, peso = self.gate_item(gnn_i, trans_i)
        if devolver_peso:
            return fusionado, peso
        return fusionado

    def estado_transformer(self, seq_entrada):
        return self.trans_model(seq_entrada)  # (batch, max_len, emb_dim)


def bpr_loss(u_emb, pos_emb, neg_emb):
    pos_scores = (u_emb * pos_emb).sum(-1)
    neg_scores = (u_emb * neg_emb).sum(-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


# ------------------------------------------------------------------
# Entrenamiento de las compuertas de fusion
# ------------------------------------------------------------------

def entrenar_fusion(nombre, epochs_ramas=20, epochs_fusion=15,
                     lr=0.001, batch_size=128,
                     freeze_ramas=True, proc_dir=PROC_DIR):
    set_seed(SEED)

    print(f"[{nombre}] (1/2) entrenando rama GNN (reutilizando modelo_gnn.py)...")
    gnn_model, gnn_edge_index, gnn_idmaps, gnn_train, gnn_val, gnn_test = \
        gnn_mod.entrenar(nombre, epochs=epochs_ramas)

    print(f"[{nombre}] (2/2) entrenando rama Transformer (reutilizando modelo_transformer.py)...")
    trans_model, idmaps, train, test, secuencias = \
        trans_mod.entrenar(nombre, epochs=epochs_ramas, proc_dir=proc_dir)

    n_users, n_items = idmaps["n_users"], idmaps["n_items"]

    with torch.no_grad():
        gnn_model.eval()
        gnn_node_emb = gnn_model(gnn_edge_index).detach()

    modelo = ModeloHibridoGate(gnn_node_emb, trans_model, n_users,
                                emb_dim=gnn_node_emb.shape[1],
                                freeze_ramas=freeze_ramas).to(DEVICE)

    parametros_entrenables = [p for p in modelo.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parametros_entrenables, lr=lr)

    secuencias_train = {u: s for u, s in secuencias.items() if len(s) >= 2}
    usuarios = list(secuencias_train.keys())

    print(f"[{nombre}] entrenando fusion con GATE ({'ramas congeladas' if freeze_ramas else 'fine-tuning completo'}): "
          f"{len(usuarios):,} usuarios, {epochs_fusion} epocas de fusion")

    modelo.train()
    if freeze_ramas:
        modelo.trans_model.eval()

    for epoch in range(1, epochs_fusion + 1):
        np.random.shuffle(usuarios)
        total_loss = 0.0
        n_batches = 0
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

            h = modelo.estado_transformer(entradas_t)  # (batch, max_len, emb_dim)
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
            print(f"[{nombre}] fusion(gate) epoca {epoch}/{epochs_fusion} - loss BPR promedio: "
                  f"{total_loss / max(1, n_batches):.4f}")

    return modelo, idmaps, train, test, secuencias


# ------------------------------------------------------------------
# Evaluacion: mismo protocolo Recall@K / NDCG@K + peso promedio del gate
# ------------------------------------------------------------------

@torch.no_grad()
def evaluar(modelo, idmaps, train, test, secuencias, k=10):
    """Multi-objetivo, restringido al catalogo de train (metricas_eval.py;
    reemplaza el protocolo anterior de 1 solo objetivo por persona).
    Conserva el peso promedio del gate como informacion adicional."""
    modelo.eval()
    n_items = idmaps["n_items"]

    secuencias_train = trans_mod.cargar_secuencias_train(train)
    catalogo = construir_catalogo(train)

    todos_items = torch.arange(n_items, dtype=torch.long, device=DEVICE)
    fused_item_all, peso_item_all = modelo.fusionar_item(todos_items, devolver_peso=True)

    usuarios_a_evaluar = test["user_idx"].unique()
    scores_por_usuario = {}
    pesos_usuario = []
    for u in usuarios_a_evaluar:
        u = int(u)
        seq = secuencias_train.get(u)
        if not seq:
            continue
        seq_tensor, longitud_real = trans_mod.secuencia_a_tensor(seq)
        entrada = torch.tensor([seq_tensor], dtype=torch.long, device=DEVICE)
        h = modelo.estado_transformer(entrada)[0, longitud_real - 1]

        u_idx_t = torch.tensor([u], dtype=torch.long, device=DEVICE)
        fused_u, peso_u = modelo.fusionar_usuario(u_idx_t, h.unsqueeze(0), devolver_peso=True)
        fused_u = fused_u[0]
        pesos_usuario.append(peso_u.item())
        scores_por_usuario[u] = (fused_item_all @ fused_u).cpu().numpy()

    resultado = evaluar_scores(scores_por_usuario, train, test, catalogo, k=k)
    resultado["peso_gate_usuario_promedio_hacia_gnn"] = (
        float(np.mean(pesos_usuario)) if pesos_usuario else 0.0)
    resultado["peso_gate_item_promedio_hacia_gnn"] = float(peso_item_all.mean().item())
    return resultado


# ------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------

def procesar(nombre, epochs_ramas=20, epochs_fusion=15, k=10,
             freeze_ramas=True, proc_dir=PROC_DIR):
    modelo, idmaps, train, test, secuencias = entrenar_fusion(
        nombre, epochs_ramas=epochs_ramas, epochs_fusion=epochs_fusion,
        freeze_ramas=freeze_ramas, proc_dir=proc_dir)

    metricas = evaluar(modelo, idmaps, train, test, secuencias, k=k)
    print(f"[{nombre}] resultados HIBRIDO+GATE en test -> {metricas}\n")

    with torch.no_grad():
        n_items = idmaps["n_items"]
        n_users = idmaps["n_users"]
        todos_usuarios = torch.arange(n_users, dtype=torch.long, device=DEVICE)
        secuencias_train = trans_mod.cargar_secuencias_train(train)
        h_por_usuario = torch.zeros(n_users, modelo.gnn_node_emb.shape[1], device=DEVICE)
        for u in range(n_users):
            seq = secuencias_train.get(u)
            if not seq:
                continue
            seq_tensor, longitud_real = trans_mod.secuencia_a_tensor(seq)
            entrada = torch.tensor([seq_tensor], dtype=torch.long, device=DEVICE)
            h_por_usuario[u] = modelo.estado_transformer(entrada)[0, longitud_real - 1]
        fused_usuarios = modelo.fusionar_usuario(todos_usuarios, h_por_usuario).cpu().numpy()
        fused_items = modelo.fusionar_item(
            torch.arange(n_items, dtype=torch.long, device=DEVICE)).cpu().numpy()

    np.save(f"{proc_dir}/{nombre}_hibrido_gate_embeddings_usuario.npy", fused_usuarios)
    np.save(f"{proc_dir}/{nombre}_hibrido_gate_embeddings_item.npy", fused_items)
    print(f"[{nombre}] embeddings fusionados (gate) guardados en "
          f"{proc_dir}/{nombre}_hibrido_gate_embeddings_*.npy "
          f"(usuarios={fused_usuarios.shape}, items={fused_items.shape})\n")
    return metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"], default="all")
    parser.add_argument("--epochs_ramas", type=int, default=20,
                         help="epocas para entrenar GNN y Transformer (si se parte de cero)")
    parser.add_argument("--epochs_fusion", type=int, default=15,
                         help="epocas para entrenar las compuertas de fusion")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fine_tune_ramas", action="store_true",
                         help="si se pasa, no se congelan las ramas: se afinan junto con la fusion")
    args = parser.parse_args()

    freeze = not args.fine_tune_ramas
    if args.dataset in ("olist", "all"):
        procesar("olist", epochs_ramas=args.epochs_ramas, epochs_fusion=args.epochs_fusion,
                  k=args.k, freeze_ramas=freeze)
    if args.dataset in ("amazon", "all"):
        procesar("amazon", epochs_ramas=args.epochs_ramas, epochs_fusion=args.epochs_fusion,
                  k=args.k, freeze_ramas=freeze)
