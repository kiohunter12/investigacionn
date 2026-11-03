"""
modelo_hibrido.py
-----------------------------------------------------------------------
Capa de fusion hibrida GNN + Transformer (Semanas 3-4 del cronograma).

Reutiliza las dos ramas YA VALIDADAS tal cual estan en modelo_gnn.py y
modelo_transformer.py (se importan sus funciones entrenar(), no se
reescribe nada de esas ramas) y les agrega una capa de fusion por
concatenacion + proyeccion lineal:

    usuario:  concat(gnn_user_emb[u], transformer_hidden_state(u)) -> Linear -> fused_user
    item:     concat(gnn_item_emb[i], transformer_item_emb[i])     -> Linear -> fused_item
    score:    fused_user . fused_item   (mismo estilo BPR que las ramas)

Por que esta forma de fusion:
- GNNEncoder.forward(edge_index) devuelve un embedding ESTATICO por nodo
  (node_emb[:n_users] = usuarios, node_emb[n_users:] = items), ver
  modelo_gnn.py.
- SecuenciaTransformer.forward(seq) devuelve un estado oculto DINAMICO
  por posicion (depende de la secuencia de cada usuario); no existe un
  "embedding de usuario" estatico en esa rama, solo item_emb (estatico,
  es lo que se guarda en *_transformer_embeddings.npy).
  => La unica representacion de usuario que tiene sentido fusionar es
     el estado oculto del Transformer en la ULTIMA posicion real de la
     secuencia (el mismo que usa modelo_transformer.evaluar()).

Por default las dos ramas se CONGELAN (freeze) y solo se entrenan las
dos capas de proyeccion de fusion: así no se destruyen los pesos ya
validados de cada rama, y la fusion es rapida de entrenar (pocos
parametros). Se puede descongelar con --fine_tune_ramas para afinar
todo el modelo junto (mas lento, mas epocas recomendadas).

Entrada: la misma que las ramas (data/processed/{nombre}_*).
Salida:
    data/processed/{nombre}_hibrido_embeddings_usuario.npy  (uno por usuario evaluado, ultimo estado)
    data/processed/{nombre}_hibrido_embeddings_item.npy     (uno por item)
    Metricas Recall@K / NDCG@K impresas en consola, mismo protocolo
    que modelo_gnn.py y modelo_transformer.py (comparten el mismo
    conjunto de test y la misma metrica -> comparables entre las tres).

Uso (desde /workspace dentro del contenedor Docker):
    python src/modelo_hibrido.py --dataset olist
    python src/modelo_hibrido.py --dataset amazon
    python src/modelo_hibrido.py --dataset all
    python src/modelo_hibrido.py --dataset olist --fine_tune_ramas
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
# Modelo de fusion: concatenacion + proyeccion lineal sobre las dos
# ramas ya entrenadas (GNNEncoder y SecuenciaTransformer)
# ------------------------------------------------------------------

class ModeloHibrido(nn.Module):
    """Envuelve las dos ramas ya entrenadas y agrega la capa de fusion.

    gnn_node_emb: tensor (n_nodes, emb_dim) PRECALCULADO una sola vez
        (es estatico, no depende de la secuencia de nadie), con
        gnn_node_emb[:n_users] = usuarios y gnn_node_emb[n_users:] = items.
    trans_model: SecuenciaTransformer ya entrenado. Se sigue usando su
        forward() para obtener el estado oculto dinamico por secuencia.
    """

    def __init__(self, gnn_node_emb, trans_model, n_users, emb_dim=64,
                 fusion_dim=64, freeze_ramas=True):
        super().__init__()
        self.register_buffer("gnn_node_emb", gnn_node_emb)
        self.trans_model = trans_model
        self.n_users = n_users

        if freeze_ramas:
            for p in self.trans_model.parameters():
                p.requires_grad = False

        # concat(gnn_emb, transformer_emb) -> fusion_dim
        self.proyeccion_usuario = nn.Linear(emb_dim * 2, fusion_dim)
        self.proyeccion_item = nn.Linear(emb_dim * 2, fusion_dim)

    def fusionar_usuario(self, u_idx, h_transformer):
        """u_idx: (N,) indices de usuario. h_transformer: (N, emb_dim)
        estado oculto del Transformer en la posicion que corresponda
        (ultima real en eval, cada posicion valida en train)."""
        gnn_u = self.gnn_node_emb[u_idx]  # (N, emb_dim) -- estatico
        fusion_in = torch.cat([gnn_u, h_transformer], dim=-1)
        return self.proyeccion_usuario(fusion_in)

    def fusionar_item(self, item_idx):
        """item_idx: (N,) indices de item (0..n_items-1, SIN +1)."""
        gnn_i = self.gnn_node_emb[self.n_users + item_idx]  # (N, emb_dim)
        trans_i = self.trans_model.item_emb(item_idx + 1)   # +1: offset de padding
        fusion_in = torch.cat([gnn_i, trans_i], dim=-1)
        return self.proyeccion_item(fusion_in)

    def estado_transformer(self, seq_entrada, devolver_todas_posiciones=False):
        """Corre la rama Transformer sobre una secuencia de entrada.
        seq_entrada: (batch, max_len) ya con padding e indices +1,
        igual que espera SecuenciaTransformer (ver modelo_transformer.py)."""
        h = self.trans_model(seq_entrada)
        return h  # (batch, max_len, emb_dim)


def bpr_loss(u_emb, pos_emb, neg_emb):
    pos_scores = (u_emb * pos_emb).sum(-1)
    neg_scores = (u_emb * neg_emb).sum(-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


# ------------------------------------------------------------------
# Entrenamiento de la capa de fusion
# ------------------------------------------------------------------

def entrenar_fusion(nombre, epochs_ramas=20, epochs_fusion=15,
                     fusion_dim=64, lr=0.001, batch_size=128,
                     freeze_ramas=True, proc_dir=PROC_DIR):
    set_seed(SEED)

    print(f"[{nombre}] (1/2) entrenando rama GNN (reutilizando modelo_gnn.py)...")
    gnn_model, gnn_edge_index, gnn_idmaps, gnn_train, gnn_val, gnn_test = \
        gnn_mod.entrenar(nombre, epochs=epochs_ramas)

    print(f"[{nombre}] (2/2) entrenando rama Transformer (reutilizando modelo_transformer.py)...")
    trans_model, idmaps, train, test, secuencias = \
        trans_mod.entrenar(nombre, epochs=epochs_ramas, proc_dir=proc_dir)

    n_users, n_items = idmaps["n_users"], idmaps["n_items"]

    # embedding GNN: estatico, se calcula UNA sola vez (no depende de
    # secuencias) y se guarda como buffer dentro del modelo hibrido
    with torch.no_grad():
        gnn_model.eval()
        gnn_node_emb = gnn_model(gnn_edge_index).detach()

    modelo = ModeloHibrido(gnn_node_emb, trans_model, n_users,
                            emb_dim=gnn_node_emb.shape[1], fusion_dim=fusion_dim,
                            freeze_ramas=freeze_ramas).to(DEVICE)

    parametros_entrenables = [p for p in modelo.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parametros_entrenables, lr=lr)

    secuencias_train = {u: s for u, s in secuencias.items() if len(s) >= 2}
    usuarios = list(secuencias_train.keys())

    print(f"[{nombre}] entrenando fusion ({'ramas congeladas' if freeze_ramas else 'fine-tuning completo'}): "
          f"{len(usuarios):,} usuarios, {epochs_fusion} epocas de fusion")

    modelo.train()
    if freeze_ramas:
        modelo.trans_model.eval()  # las ramas congeladas no acumulan estadisticas de dropout/BN

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

            # repetir el usuario en cada posicion valida de su secuencia
            u_idx_expandido = u_idx_t.unsqueeze(1).expand(-1, h.shape[1])

            h_v = h[mask_validos]
            u_idx_v = u_idx_expandido[mask_validos]
            pos_v = (objetivos_t[mask_validos] - 1)  # quitar offset +1 -> item_idx real
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
            print(f"[{nombre}] fusion epoca {epoch}/{epochs_fusion} - loss BPR promedio: "
                  f"{total_loss / max(1, n_batches):.4f}")

    return modelo, idmaps, train, test, secuencias


# ------------------------------------------------------------------
# Evaluacion: mismo protocolo Recall@K / NDCG@K que las otras ramas
# ------------------------------------------------------------------

@torch.no_grad()
def evaluar(modelo, idmaps, train, test, secuencias, k=10):
    """Multi-objetivo, restringido al catalogo de train (metricas_eval.py;
    reemplaza el protocolo anterior de 1 solo objetivo por persona)."""
    modelo.eval()
    n_items = idmaps["n_items"]

    secuencias_train = trans_mod.cargar_secuencias_train(train)
    catalogo = construir_catalogo(train)

    todos_items = torch.arange(n_items, dtype=torch.long, device=DEVICE)
    fused_item_all = modelo.fusionar_item(todos_items)  # (n_items, fusion_dim)

    usuarios_a_evaluar = test["user_idx"].unique()
    scores_por_usuario = {}
    for u in usuarios_a_evaluar:
        u = int(u)
        seq = secuencias_train.get(u)
        if not seq:
            continue
        seq_tensor, longitud_real = trans_mod.secuencia_a_tensor(seq)
        entrada = torch.tensor([seq_tensor], dtype=torch.long, device=DEVICE)
        h = modelo.estado_transformer(entrada)[0, longitud_real - 1]  # ultimo estado real

        u_idx_t = torch.tensor([u], dtype=torch.long, device=DEVICE)
        fused_u = modelo.fusionar_usuario(u_idx_t, h.unsqueeze(0))[0]  # (fusion_dim,)
        scores_por_usuario[u] = (fused_item_all @ fused_u).cpu().numpy()

    return evaluar_scores(scores_por_usuario, train, test, catalogo, k=k)


# ------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------

def procesar(nombre, epochs_ramas=20, epochs_fusion=15, k=10,
             freeze_ramas=True, proc_dir=PROC_DIR):
    modelo, idmaps, train, test, secuencias = entrenar_fusion(
        nombre, epochs_ramas=epochs_ramas, epochs_fusion=epochs_fusion,
        freeze_ramas=freeze_ramas, proc_dir=proc_dir)

    metricas = evaluar(modelo, idmaps, train, test, secuencias, k=k)
    print(f"[{nombre}] resultados HIBRIDO en test (principal/secundario) -> {metricas}\n")

    with torch.no_grad():
        n_items = idmaps["n_items"]
        n_users = idmaps["n_users"]
        todos_usuarios = torch.arange(n_users, dtype=torch.long, device=DEVICE)
        # estado "usuario" generico: ultimo estado observado en train (si existe)
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

    np.save(f"{proc_dir}/{nombre}_hibrido_embeddings_usuario.npy", fused_usuarios)
    np.save(f"{proc_dir}/{nombre}_hibrido_embeddings_item.npy", fused_items)
    print(f"[{nombre}] embeddings fusionados guardados en {proc_dir}/{nombre}_hibrido_embeddings_*.npy "
          f"(usuarios={fused_usuarios.shape}, items={fused_items.shape})\n")
    return metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"], default="all")
    parser.add_argument("--epochs_ramas", type=int, default=20,
                         help="epocas para entrenar GNN y Transformer (si se parte de cero)")
    parser.add_argument("--epochs_fusion", type=int, default=15,
                         help="epocas para entrenar las capas de proyeccion de fusion")
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