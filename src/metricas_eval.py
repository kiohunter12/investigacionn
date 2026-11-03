"""
metricas_eval.py
-----------------------------------------------------------------------
Modulo compartido de evaluacion, exactamente segun la Seccion IV-D
(Protocolo de evaluacion) del short paper v2:

Para cada persona u: H_u = productos de entrenamiento, T_u = productos
objetivo (puede tener mas de uno, porque el objetivo es el MOMENTO DE
COMPRA completo, no una sola fila), y L_u = los 10 productos con mayor
puntaje entre los del CATALOGO (items con >=1 interaccion de
entrenamiento) que no estan en H_u, sin muestreo de negativos.

    Recall@10 = (1/N) sum_u |T_u ∩ L_u| / |T_u|
    HR@10     = (1/N) sum_u 1[T_u ∩ L_u != vacio]
    NDCG@10   = (1/N) sum_u ( sum_{i in T_u∩L_u} 1/log2(r_ui+1) )
                        / ( sum_{j=1}^{min(|T_u|,10)} 1/log2(j+1) )

Con un unico objetivo, Recall@10 y HR@10 coinciden (caso del protocolo
anterior, ya superado).

Reemplaza al `evaluar()` de modelo_gnn.py / modelo_gnn_5semillas.py /
modelo_transformer.py / modelo_baselines.py, que:
  (a) solo guardaba 1 item objetivo por usuario
      (`test.set_index("user_idx")["item_idx"].to_dict()` pisaba los
      demas si una persona tenia 2+ productos en su ultimo pedido), y
  (b) puntuaba sobre TODOS los items codificados (incluidos los que
      solo aparecen en val/test y nunca se entrenaron, con embeddings
      aleatorios que podian rankear alto por ruido) en vez de
      restringir al catalogo de entrenamiento.

Reporta DOS analisis, tal como describe la Seccion IV-B/IV-D (arranque
en frio, OE4):
  - "principal": solo personas con >=1 objetivo DENTRO del catalogo,
    y dentro de esas personas, T_u se restringe a los objetivos en
    catalogo (los fuera de catalogo de esa misma persona no entran ni
    al numerador ni al denominador). En Olist: 869 personas / 926
    objetivos.
  - "secundario": TODAS las personas con objetivo, T_u completo
    (incluye objetivos fuera de catalogo, que nunca pueden aparecer en
    L_u y por lo tanto siempre cuentan como fallo). En Olist: 1147
    personas / 1242 objetivos.
-----------------------------------------------------------------------
"""
import numpy as np


def construir_catalogo(train_df):
    """Items con >=1 interaccion de entrenamiento (candidatos validos)."""
    return np.sort(train_df["item_idx"].unique().astype(np.int64))


def construir_objetivos(eval_df):
    """T_u por persona: conjunto de item_idx objetivo (1 o mas)."""
    return eval_df.groupby("user_idx")["item_idx"].apply(
        lambda s: set(s.astype(int))).to_dict()


def construir_vistos(train_df):
    """H_u por persona: conjunto de item_idx ya comprados en train."""
    return train_df.groupby("user_idx")["item_idx"].apply(
        lambda s: set(s.astype(int))).to_dict()


def _top_k_catalogo(scores, catalogo, vistos, k=10):
    """L_u: los k productos de mayor puntaje en catalogo \\ vistos."""
    if vistos:
        mask = np.isin(catalogo, list(vistos), invert=True)
        candidatos = catalogo[mask]
    else:
        candidatos = catalogo
    if len(candidatos) == 0:
        return np.array([], dtype=np.int64)
    k_ef = min(k, len(candidatos))
    sub = scores[candidatos]
    top_local = np.argpartition(-sub, k_ef - 1)[:k_ef]
    top_local = top_local[np.argsort(-sub[top_local])]
    return candidatos[top_local]


def _metricas_usuario(L_u, objetivo_u, k=10):
    """Recall@10, HR@10, NDCG@10 para una persona, formulas exactas IV-D."""
    if not objetivo_u:
        return None
    hits_rank = [(rank, item) for rank, item in enumerate(L_u, start=1)
                 if item in objetivo_u]
    recall = len(hits_rank) / len(objetivo_u)
    hr = 1.0 if hits_rank else 0.0
    dcg = sum(1.0 / np.log2(rank + 1) for rank, _ in hits_rank)
    idcg_n = min(len(objetivo_u), k)
    idcg = sum(1.0 / np.log2(j + 1) for j in range(1, idcg_n + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, hr, ndcg


def evaluar_scores(scores_por_usuario, train_df, eval_df, catalogo, k=10):
    """scores_por_usuario: dict user_idx -> np.array de puntajes,
    indexado por item_idx, de longitud n_items (cualquier valor fuera
    del catalogo se ignora porque _top_k_catalogo ya restringe a
    `catalogo`).

    Devuelve {"principal": {...}, "secundario": {...}} con
    recall@k, hr@k, ndcg@k, n_personas, n_objetivos en cada uno.
    """
    vistos_por_usuario = construir_vistos(train_df)
    objetivos_por_usuario = construir_objetivos(eval_df)
    catalogo_set = set(catalogo.tolist())

    principal_r, principal_h, principal_n = [], [], []
    secundario_r, secundario_h, secundario_n = [], [], []
    n_obj_principal = 0
    n_obj_secundario = 0

    for u, objetivo_u in objetivos_por_usuario.items():
        scores_u = scores_por_usuario.get(u)
        vistos_u = vistos_por_usuario.get(u, set())
        if scores_u is None:
            L_u = np.array([], dtype=np.int64)
        else:
            L_u = _top_k_catalogo(scores_u, catalogo, vistos_u, k=k)

        # --- secundario: T_u completo (incluye fuera de catalogo) ---
        m_sec = _metricas_usuario(L_u, objetivo_u, k=k)
        if m_sec is not None:
            r, h, n = m_sec
            secundario_r.append(r); secundario_h.append(h); secundario_n.append(n)
            n_obj_secundario += len(objetivo_u)

        # --- principal: solo objetivos dentro de catalogo ---
        objetivo_u_cat = objetivo_u & catalogo_set
        if objetivo_u_cat:
            m_pri = _metricas_usuario(L_u, objetivo_u_cat, k=k)
            r, h, n = m_pri
            principal_r.append(r); principal_h.append(h); principal_n.append(n)
            n_obj_principal += len(objetivo_u_cat)

    def resumen(rs, hs, ns, n_pers, n_obj):
        return {
            "n_personas": n_pers,
            "n_objetivos": n_obj,
            f"recall@{k}": float(np.mean(rs)) if rs else 0.0,
            f"hr@{k}": float(np.mean(hs)) if hs else 0.0,
            f"ndcg@{k}": float(np.mean(ns)) if ns else 0.0,
        }

    return {
        "principal": resumen(principal_r, principal_h, principal_n,
                              len(principal_r), n_obj_principal),
        "secundario": resumen(secundario_r, secundario_h, secundario_n,
                               len(secundario_r), n_obj_secundario),
    }


def evaluar_scores_detallado(scores_por_usuario, train_df, eval_df, catalogo, k=10):
    """Igual que evaluar_scores, pero ADEMAS devuelve, por persona, su
    HR@10 y NDCG@10 individuales (0/1 y valor real), indexados por
    user_idx. Se usa para las pruebas pareadas (McNemar sobre HR@10,
    bootstrap pareado sobre NDCG@10, Seccion IV-F) que necesitan el
    resultado de CADA persona, no solo el promedio."""
    vistos_por_usuario = construir_vistos(train_df)
    objetivos_por_usuario = construir_objetivos(eval_df)
    catalogo_set = set(catalogo.tolist())

    hr_principal, ndcg_principal = {}, {}
    hr_secundario, ndcg_secundario = {}, {}

    for u, objetivo_u in objetivos_por_usuario.items():
        scores_u = scores_por_usuario.get(u)
        vistos_u = vistos_por_usuario.get(u, set())
        if scores_u is None:
            L_u = np.array([], dtype=np.int64)
        else:
            L_u = _top_k_catalogo(scores_u, catalogo, vistos_u, k=k)

        m_sec = _metricas_usuario(L_u, objetivo_u, k=k)
        if m_sec is not None:
            _, h, n = m_sec
            hr_secundario[u] = h
            ndcg_secundario[u] = n

        objetivo_u_cat = objetivo_u & catalogo_set
        if objetivo_u_cat:
            m_pri = _metricas_usuario(L_u, objetivo_u_cat, k=k)
            _, h, n = m_pri
            hr_principal[u] = h
            ndcg_principal[u] = n

    return {
        "principal": {"hr": hr_principal, "ndcg": ndcg_principal},
        "secundario": {"hr": hr_secundario, "ndcg": ndcg_secundario},
    }
