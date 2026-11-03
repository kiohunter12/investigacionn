"""
analisis_estadistico.py
-----------------------------------------------------------------------
Pruebas de contraste de la Seccion IV-F del short paper:
  - McNemar (exacto, binomial) sobre HR@10 pareado por persona, para
    OE1 (mejor fusion vs GNN vs Transformer), OE2 (fusiones entre si)
    y OE3 (todas las configuraciones vs los 3 baselines).
  - Bootstrap pareado (diferencia de NDCG@10) con IC 95%.
  - Correccion de Holm para comparaciones multiples.

Las 5 semillas se agrupan en una sola prueba por configuracion
(decision explicita del usuario): cada persona aparece hasta 5 veces
(una por semilla) como observaciones pareadas, N efectivo ~= 5 x 1147.
CAVEAT que se reporta junto a los resultados: Popularidad e ItemKNN no
tienen variacion entre semillas (son deterministicos), asi que su
"5 replicas" son la MISMA salida repetida 5 veces -- no agregan
informacion independiente nueva en ese lado de la comparacion. Se
mantiene asi solo porque fue la opcion elegida explicitamente; en
sentido estricto la literatura recomendaria para esos pares un McNemar
simple de 1147 personas (N=1), no el pool de 5735.

Reutiliza directamente los artefactos YA ENTRENADOS (embeddings/pesos
guardados por modelo_gnn_5semillas.py y modelo_transformer_5semillas.py)
para GNN, Transformer, Popularidad e ItemKNN -- no hace falta
reentrenar nada para esos cuatro. BPR-MF y las 3 fusiones SI necesitan
reentrenarse una vez mas (sus pesos no se guardaron la primera vez),
pero es rapido (ya se verifico antes: minutos, no horas).

Salida:
    results/analisis_estadistico_mcnemar_bootstrap.csv
-----------------------------------------------------------------------
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

from metricas_eval import construir_catalogo, evaluar_scores_detallado
from modelo_transformer import SecuenciaTransformer, MAX_LEN
from modelo_transformer_5semillas import cargar_secuencias_train
from modelo_baselines_simples import (
    scores_popularidad, construir_matriz_usuario_item,
    similitud_coseno_items, scores_itemknn_usuarios, entrenar_bprmf,
)
from modelo_fusion_5semillas import entrenar_fusion, evaluar as evaluar_fusion

PROC_DIR = "data/processed"
SEEDS = [42, 123, 2024, 2025, 2026]
K = 10


def cargar_base(nombre):
    idmaps = json.load(open(f"{PROC_DIR}/{nombre}_id_maps.json"))
    train = pd.read_csv(f"{PROC_DIR}/{nombre}_train.csv")
    val = pd.read_csv(f"{PROC_DIR}/{nombre}_val.csv")
    test = pd.read_csv(f"{PROC_DIR}/{nombre}_test.csv")
    catalogo = construir_catalogo(train)
    return idmaps, train, val, test, catalogo


# ------------------------------------------------------------------
# Per-config, per-seed: hr/ndcg POR PERSONA en el analisis "principal"
# ------------------------------------------------------------------

def hits_gnn(nombre, seed, train, test, catalogo, idmaps):
    n_users = idmaps["n_users"]
    node_emb = np.load(f"{PROC_DIR}/{nombre}_gnn_embeddings_seed{seed}.npy")
    user_emb, item_emb = node_emb[:n_users], node_emb[n_users:]
    scores = {int(u): user_emb[int(u)] @ item_emb.T for u in test["user_idx"].unique()}
    det = evaluar_scores_detallado(scores, train, test, catalogo, k=K)
    return det["principal"]["hr"], det["principal"]["ndcg"]


def hits_transformer(nombre, seed, train, test, catalogo, idmaps):
    n_items = idmaps["n_items"]
    model = SecuenciaTransformer(n_items, max_len=MAX_LEN, emb_dim=64)
    model.load_state_dict(torch.load(f"{PROC_DIR}/{nombre}_transformer_modelo_seed{seed}.pt",
                                      map_location="cpu"))
    model.eval()
    secuencias_train = cargar_secuencias_train(train)
    item_emb_all = model.item_emb.weight[1:].detach()
    from modelo_transformer import secuencia_a_tensor
    scores = {}
    with torch.no_grad():
        for u in test["user_idx"].unique():
            u = int(u)
            seq = secuencias_train.get(u)
            if not seq:
                continue
            seq_t, long_real = secuencia_a_tensor(seq)
            h = model(torch.tensor([seq_t], dtype=torch.long))[0, long_real - 1]
            scores[u] = (item_emb_all @ h).numpy()
    det = evaluar_scores_detallado(scores, train, test, catalogo, k=K)
    return det["principal"]["hr"], det["principal"]["ndcg"]


def hits_popularidad(nombre, train, test, catalogo, idmaps):
    n_items = idmaps["n_items"]
    vec = scores_popularidad(train, n_items)
    scores = {int(u): vec for u in test["user_idx"].unique()}
    det = evaluar_scores_detallado(scores, train, test, catalogo, k=K)
    return det["principal"]["hr"], det["principal"]["ndcg"]


def hits_itemknn(nombre, train, test, catalogo, idmaps, n_vecinos=50):
    n_users, n_items = idmaps["n_users"], idmaps["n_items"]
    m = construir_matriz_usuario_item(train, n_users, n_items)
    sim = similitud_coseno_items(m, n_vecinos=n_vecinos)
    scores = scores_itemknn_usuarios(m, sim, test["user_idx"].unique())
    det = evaluar_scores_detallado(scores, train, test, catalogo, k=K)
    return det["principal"]["hr"], det["principal"]["ndcg"]


def hits_bprmf(nombre, seed, train, val, test, catalogo, idmaps):
    model, _, _ = entrenar_bprmf(nombre, train, val, idmaps, epochs=50, patience=5,
                                  seed=seed, k=K)
    with torch.no_grad():
        user_emb = model.user_emb.weight.numpy()
        item_emb = model.item_emb.weight.numpy()
    scores = {int(u): user_emb[int(u)] @ item_emb.T for u in test["user_idx"].unique()}
    det = evaluar_scores_detallado(scores, train, test, catalogo, k=K)
    return det["principal"]["hr"], det["principal"]["ndcg"]


def hits_fusion(nombre, tipo, seed):
    fusion, idmaps, gnn_node_emb, trans_model, train, val, test, _, _ = entrenar_fusion(
        nombre, tipo, seed, epochs=20, patience=5, k=K)
    import torch as T

    @T.no_grad()
    def evaluar_detallado_fusion():
        secuencias_train = cargar_secuencias_train(train)
        catalogo = construir_catalogo(train)
        n_users, n_items = idmaps["n_users"], idmaps["n_items"]
        todos_items = T.arange(n_items, dtype=T.long)
        gnn_items = gnn_node_emb[n_users + todos_items]
        trans_items = trans_model.item_emb(todos_items + 1)
        fused_item_all = fusion.item(gnn_items, trans_items)
        from modelo_fusion_5semillas import estado_transformer_usuario
        scores = {}
        for u in test["user_idx"].unique():
            u = int(u)
            h = estado_transformer_usuario(trans_model, secuencias_train, u)
            if h is None:
                continue
            gnn_u = gnn_node_emb[u].unsqueeze(0)
            fused_u = fusion.usuario(gnn_u, h.unsqueeze(0))[0]
            scores[u] = (fused_item_all @ fused_u).numpy()
        return evaluar_scores_detallado(scores, train, test, catalogo, k=K)

    det = evaluar_detallado_fusion()
    return det["principal"]["hr"], det["principal"]["ndcg"]


# ------------------------------------------------------------------
# Pruebas pareadas
# ------------------------------------------------------------------

def mcnemar_pareado(hr_a_por_seed, hr_b_por_seed):
    """hr_a_por_seed / hr_b_por_seed: listas de dicts {persona: 0/1},
    una por semilla (o una sola, para configs deterministicas, que se
    repite para alinear el pool). Se agrupan todas las semillas en un
    solo McNemar exacto (binomial sobre discordantes)."""
    b = c = n_concordante = 0
    personas_comunes_total = 0
    for hr_a, hr_b in zip(hr_a_por_seed, hr_b_por_seed):
        personas = set(hr_a.keys()) & set(hr_b.keys())
        personas_comunes_total += len(personas)
        for p in personas:
            a, bb = hr_a[p], hr_b[p]
            if a == 1 and bb == 0:
                b += 1
            elif a == 0 and bb == 1:
                c += 1
            else:
                n_concordante += 1
    n_discordante = b + c
    if n_discordante == 0:
        p_valor = 1.0
    else:
        p_valor = sps.binomtest(min(b, c), n_discordante, 0.5).pvalue
    return {
        "n_pares": personas_comunes_total, "a_mayor_b": b, "b_mayor_a": c,
        "concordantes": n_concordante, "p_valor_mcnemar": p_valor,
    }


def bootstrap_pareado_ndcg(ndcg_a_por_seed, ndcg_b_por_seed, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    diffs_obs = []
    pool = []
    for ndcg_a, ndcg_b in zip(ndcg_a_por_seed, ndcg_b_por_seed):
        personas = sorted(set(ndcg_a.keys()) & set(ndcg_b.keys()))
        for p in personas:
            pool.append(ndcg_a[p] - ndcg_b[p])
    pool = np.array(pool)
    diff_media = pool.mean()
    boot_medias = np.array([
        rng.choice(pool, size=len(pool), replace=True).mean() for _ in range(n_boot)])
    ic_low, ic_high = np.percentile(boot_medias, [2.5, 97.5])
    return {"n_pares": len(pool), "diff_ndcg_media": float(diff_media),
            "ic95_low": float(ic_low), "ic95_high": float(ic_high)}


def holm(p_valores):
    """Correccion de Holm: devuelve p-valores ajustados, mismo orden
    de entrada."""
    idx_orden = np.argsort(p_valores)
    m = len(p_valores)
    ajustados = np.empty(m)
    maximo = 0.0
    for rank, idx in enumerate(idx_orden):
        ajustado = (m - rank) * p_valores[idx]
        maximo = max(maximo, ajustado)
        ajustados[idx] = min(maximo, 1.0)
    return ajustados


# ------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------

def main(nombre="olist"):
    idmaps, train, val, test, catalogo = cargar_base(nombre)

    print("[1/3] recolectando hits por persona para cada configuracion...")
    hr_todas, ndcg_todas = {}, {}

    # --- deterministicas: 1 sola corrida real, repetida para alinear ---
    hr_pop, ndcg_pop = hits_popularidad(nombre, train, test, catalogo, idmaps)
    hr_todas["popularidad"] = [hr_pop] * len(SEEDS)
    ndcg_todas["popularidad"] = [ndcg_pop] * len(SEEDS)
    print("  popularidad OK (deterministico, 1 corrida real)")

    hr_knn, ndcg_knn = hits_itemknn(nombre, train, test, catalogo, idmaps)
    hr_todas["itemknn"] = [hr_knn] * len(SEEDS)
    ndcg_todas["itemknn"] = [ndcg_knn] * len(SEEDS)
    print("  itemknn OK (deterministico, 1 corrida real)")

    # --- estocasticas: ya entrenadas, solo reevaluar con artefactos guardados ---
    hr_todas["gnn"], ndcg_todas["gnn"] = [], []
    for seed in SEEDS:
        h, n = hits_gnn(nombre, seed, train, test, catalogo, idmaps)
        hr_todas["gnn"].append(h); ndcg_todas["gnn"].append(n)
    print("  gnn OK (5 semillas, embeddings reusados)")

    hr_todas["transformer"], ndcg_todas["transformer"] = [], []
    for seed in SEEDS:
        h, n = hits_transformer(nombre, seed, train, test, catalogo, idmaps)
        hr_todas["transformer"].append(h); ndcg_todas["transformer"].append(n)
    print("  transformer OK (5 semillas, pesos reusados)")

    # --- necesitan reentrenar (pesos no se guardaron la 1a vez) ---
    hr_todas["bprmf"], ndcg_todas["bprmf"] = [], []
    for seed in SEEDS:
        h, n = hits_bprmf(nombre, seed, train, val, test, catalogo, idmaps)
        hr_todas["bprmf"].append(h); ndcg_todas["bprmf"].append(n)
    print("  bprmf OK (5 semillas, reentrenado)")

    for tipo in ("concat", "gate", "suma"):
        hr_todas[f"fusion_{tipo}"], ndcg_todas[f"fusion_{tipo}"] = [], []
        for seed in SEEDS:
            h, n = hits_fusion(nombre, tipo, seed)
            hr_todas[f"fusion_{tipo}"].append(h); ndcg_todas[f"fusion_{tipo}"].append(n)
        print(f"  fusion_{tipo} OK (5 semillas, reentrenado)")

    os.makedirs("results", exist_ok=True)
    np.save("results/_hr_todas_por_seed.npy", hr_todas, allow_pickle=True)
    np.save("results/_ndcg_todas_por_seed.npy", ndcg_todas, allow_pickle=True)

    # --- comparaciones (OE1, OE2, OE3) ---
    # mejor fusion por NDCG@10 promedio (se decide con lo ya calculado)
    media_ndcg_fusion = {t: np.mean([np.mean(list(d.values())) for d in ndcg_todas[f"fusion_{t}"]])
                          for t in ("concat", "gate", "suma")}
    mejor_fusion = max(media_ndcg_fusion, key=media_ndcg_fusion.get)
    print(f"\n[2/3] mejor fusion por NDCG@10 (principal): {mejor_fusion} {media_ndcg_fusion}")

    comparaciones = [
        ("OE1", f"fusion_{mejor_fusion}", "gnn"),
        ("OE1", f"fusion_{mejor_fusion}", "transformer"),
        ("OE2", "fusion_concat", "fusion_gate"),
        ("OE2", "fusion_concat", "fusion_suma"),
        ("OE2", "fusion_gate", "fusion_suma"),
        ("OE3", "itemknn", f"fusion_{mejor_fusion}"),
        ("OE3", "itemknn", "gnn"),
        ("OE3", "popularidad", f"fusion_{mejor_fusion}"),
        ("OE3", "bprmf", f"fusion_{mejor_fusion}"),
    ]

    print("[3/3] corriendo McNemar + bootstrap pareado para cada comparacion...")
    filas = []
    for grupo, a, b in comparaciones:
        mcn = mcnemar_pareado(hr_todas[a], hr_todas[b])
        boot = bootstrap_pareado_ndcg(ndcg_todas[a], ndcg_todas[b])
        filas.append({"grupo": grupo, "config_a": a, "config_b": b, **mcn, **boot})

    tabla = pd.DataFrame(filas)
    tabla["p_valor_holm"] = holm(tabla["p_valor_mcnemar"].to_numpy())
    tabla["significativo_alpha_05"] = tabla["p_valor_holm"] < 0.05
    tabla.to_csv(f"results/{nombre}_analisis_estadistico.csv", index=False)
    print(f"\nguardado: results/{nombre}_analisis_estadistico.csv")
    print(tabla.to_string(index=False))
    return tabla


if __name__ == "__main__":
    main("olist")
