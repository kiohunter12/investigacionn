"""
modelo_xai.py
-----------------------------------------------------------------------
Modulo de explicabilidad (XAI) sobre el modelo hibrido con compuerta
(modelo_hibrido_gate.py), via LIME adaptado a recomendacion secuencial.

Que explica:
Para un usuario y el item que el hibrido recomienda en el top-1 (el de
mayor score, excluyendo items ya vistos), LIME responde: "de las
interacciones pasadas de este usuario, cuales pesaron mas para que el
modelo recomendara este item?"

Adaptacion de LIME (Ribeiro et al., 2016) a este problema -- no es un
dataset tabular clasico, es una secuencia de interacciones pasadas:

- Representacion interpretable: vector binario z en {0,1}^n, uno por
  cada interaccion pasada real del usuario (ventana de hasta MAX_LEN,
  la misma que usa el Transformer), donde 1 = "esa interaccion sigue
  presente en la secuencia", 0 = "se enmascara (se quita) antes de
  pasarla al Transformer".
- Vecindad local: se generan N perturbaciones z_i apagando cada
  interaccion con probabilidad 0.5 (muestreo estandar de LIME tabular),
  siempre incluyendo el punto original (todo encendido).
- Prediccion de la "caja negra" (el modelo hibrido completo) para cada
  perturbacion: se recalcula el estado oculto del Transformer con la
  secuencia enmascarada, se fusiona con la rama GNN via el gate ya
  entrenado, y se obtiene el score fusionado para el item recomendado
  -- el mismo score que el hibrido usa para rankear.
- Modelo local interpretable: regresion RIDGE lineal ponderada por un
  kernel de proximidad (distancia coseno entre z_i y "todo presente"),
  igual que el LIME original. Los coeficientes son el peso de
  importancia que LIME asigna a cada interaccion pasada para esa
  recomendacion especifica (positivo = la sostiene, negativo = la
  contradice).
- Fidelidad de la explicacion: R^2 ponderado del ajuste del modelo
  local sobre las muestras perturbadas -- que tan bien el modelo
  lineal local aproxima el comportamiento real del hibrido en esa
  vecindad. Es la metrica estandar de fidelidad de LIME.

Este modulo es POST-HOC: no modifica los pesos ni el ranking del
hibrido, solo lo explica despues de la prediccion. Por diseno, esto
garantiza que las metricas de ranking (Recall@10/NDCG@10) son
IDENTICAS con o sin el modulo XAI activado -- responde directamente a
H2 del PPI (el modulo XAI no degrada el desempeno tecnico) sin
necesidad de reentrenar nada.

Se complementa con los otros dos insumos de interpretabilidad ya
existentes en el proyecto:
    - pesos del Selector (modelo_transformer.py): relevancia de cada
      interaccion DENTRO del encoder Transformer.
    - peso del gate (modelo_hibrido_gate.py): cuanto confia la fusion
      en la rama GNN vs. la rama Transformer, por usuario/item.
    - peso LIME (este script): cuanto pesa cada interaccion pasada en
      la recomendacion FINAL del sistema completo.

Entrada: reutiliza entrenar_fusion() de modelo_hibrido_gate.py (no se
reescribe nada de las ramas ni de la fusion).
Salida:
    Explicacion impresa en consola para una muestra de usuarios de
    test (interaccion -> peso LIME, ordenadas por |peso|), y la
    fidelidad (R^2) promedio.
    data/processed/{nombre}_xai_fidelidad.json con el detalle.

Uso (desde /workspace dentro del contenedor Docker):
    python src/modelo_xai.py --dataset olist --n_usuarios 10
    python src/modelo_xai.py --dataset amazon --n_usuarios 10
    python src/modelo_xai.py --dataset all
-----------------------------------------------------------------------
"""
import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import Ridge

import modelo_transformer as trans_mod
from modelo_hibrido_gate import entrenar_fusion, DEVICE, PROC_DIR, SEED

MAX_LEN = trans_mod.MAX_LEN


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------
# Funcion "caja negra": score fusionado del hibrido para un item,
# dada una secuencia de interacciones pasadas PERTURBADA (algunas
# apagadas segun el vector binario z).
# ------------------------------------------------------------------

@torch.no_grad()
def score_item_para_secuencia(modelo, u_idx, z, seq_items, item_objetivo):
    seq_perturbada = [it for it, on in zip(seq_items, z) if on == 1]
    if len(seq_perturbada) == 0:
        seq_perturbada = [seq_items[-1]]  # necesita al menos 1 interaccion real

    seq_tensor, longitud_real = trans_mod.secuencia_a_tensor(seq_perturbada)
    entrada = torch.tensor([seq_tensor], dtype=torch.long, device=DEVICE)
    h = modelo.estado_transformer(entrada)[0, longitud_real - 1]

    u_idx_t = torch.tensor([u_idx], dtype=torch.long, device=DEVICE)
    fused_u = modelo.fusionar_usuario(u_idx_t, h.unsqueeze(0))[0]

    item_t = torch.tensor([item_objetivo], dtype=torch.long, device=DEVICE)
    fused_item = modelo.fusionar_item(item_t)[0]

    return float((fused_u @ fused_item).item())


# ------------------------------------------------------------------
# LIME adaptado a secuencias de interacciones
# ------------------------------------------------------------------

def explicar_lime(modelo, u_idx, seq_items, item_objetivo, n_muestras=200, kernel_width=0.5):
    """Devuelve (lista de (item, peso_lime), fidelidad_r2)."""
    n = len(seq_items)
    if n == 0:
        return [], 0.0

    todo_encendido = np.ones(n)

    Z = np.random.binomial(1, 0.5, size=(n_muestras, n)).astype(float)
    Z[0] = todo_encendido  # siempre incluir el punto original (sin perturbar)

    y = np.array([
        score_item_para_secuencia(modelo, u_idx, Z[i], seq_items, item_objetivo)
        for i in range(n_muestras)
    ])

    # kernel de proximidad LIME: distancia coseno a "todo encendido"
    norm_z = np.linalg.norm(Z, axis=1)
    norm_full = np.linalg.norm(todo_encendido)
    distancias = 1.0 - (Z @ todo_encendido) / (norm_z * norm_full + 1e-8)
    pesos_kernel = np.exp(-(distancias ** 2) / (kernel_width ** 2))

    modelo_local = Ridge(alpha=1.0)
    modelo_local.fit(Z, y, sample_weight=pesos_kernel)

    y_pred = modelo_local.predict(Z)
    media_pond = np.average(y, weights=pesos_kernel)
    ss_res = np.sum(pesos_kernel * (y - y_pred) ** 2)
    ss_tot = np.sum(pesos_kernel * (y - media_pond) ** 2)
    fidelidad_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0

    pesos_interaccion = list(zip(seq_items, modelo_local.coef_))
    return pesos_interaccion, fidelidad_r2


# ------------------------------------------------------------------
# Orquestacion: entrena/reutiliza el hibrido y explica una muestra
# de usuarios de test
# ------------------------------------------------------------------

def procesar(nombre, epochs_ramas=50, epochs_fusion=15, n_usuarios=10,
             n_muestras=200, proc_dir=PROC_DIR):
    set_seed(SEED)
    modelo, idmaps, train, test, secuencias = entrenar_fusion(
        nombre, epochs_ramas=epochs_ramas, epochs_fusion=epochs_fusion,
        freeze_ramas=True, proc_dir=proc_dir)
    modelo.eval()

    secuencias_train = trans_mod.cargar_secuencias_train(train)
    items_test = test.set_index("user_idx")["item_idx"].to_dict()

    usuarios_candidatos = [u for u in items_test
                            if secuencias_train.get(u) and len(secuencias_train[u]) >= 2]
    np.random.shuffle(usuarios_candidatos)
    usuarios_muestra = usuarios_candidatos[:n_usuarios]

    n_items = idmaps["n_items"]
    resultados = []

    print(f"\n[{nombre}] explicando {len(usuarios_muestra)} usuarios con LIME "
          f"({n_muestras} muestras por usuario)...\n")

    for u in usuarios_muestra:
        seq = secuencias_train[u][-MAX_LEN:]

        with torch.no_grad():
            seq_tensor, longitud_real = trans_mod.secuencia_a_tensor(seq)
            entrada = torch.tensor([seq_tensor], dtype=torch.long, device=DEVICE)
            h = modelo.estado_transformer(entrada)[0, longitud_real - 1]
            u_idx_t = torch.tensor([u], dtype=torch.long, device=DEVICE)
            fused_u = modelo.fusionar_usuario(u_idx_t, h.unsqueeze(0))[0]
            todos_items = torch.arange(n_items, dtype=torch.long, device=DEVICE)
            fused_items = modelo.fusionar_item(todos_items)
            scores = (fused_items @ fused_u).cpu().numpy()
            vistos = set(seq)
            if vistos:
                scores[list(vistos)] = -np.inf
            item_top1 = int(np.argmax(scores))

        pesos, fidelidad = explicar_lime(modelo, u, seq, item_top1, n_muestras=n_muestras)
        pesos_ordenados = sorted(pesos, key=lambda x: -abs(x[1]))

        print(f"[{nombre}] usuario {u} -> recomendado item {item_top1} "
              f"(fidelidad LIME R^2={fidelidad:.3f})")
        for item, peso in pesos_ordenados[:5]:
            signo = "+" if peso >= 0 else "-"
            print(f"    interaccion con item {item}: peso LIME {signo}{abs(peso):.4f}")

        resultados.append({
            "usuario": u,
            "item_recomendado": item_top1,
            "fidelidad_r2": fidelidad,
            "pesos_top5": [(int(it), float(p)) for it, p in pesos_ordenados[:5]],
        })

    fidelidad_promedio = float(np.mean([r["fidelidad_r2"] for r in resultados])) if resultados else 0.0
    print(f"\n[{nombre}] fidelidad LIME promedio (R^2) sobre {len(resultados)} usuarios: "
          f"{fidelidad_promedio:.4f}\n")

    resumen = {
        "dataset": nombre,
        "n_usuarios_explicados": len(resultados),
        "n_muestras_lime_por_usuario": n_muestras,
        "fidelidad_r2_promedio": fidelidad_promedio,
        "detalle": resultados,
    }
    with open(f"{proc_dir}/{nombre}_xai_fidelidad.json", "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"[{nombre}] resumen guardado en {proc_dir}/{nombre}_xai_fidelidad.json\n")
    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["olist", "amazon", "all"], default="all")
    parser.add_argument("--epochs_ramas", type=int, default=50,
                         help="debe coincidir con lo usado en modelo_hibrido_gate.py "
                              "para reportar sobre el mismo modelo que el paper")
    parser.add_argument("--epochs_fusion", type=int, default=15)
    parser.add_argument("--n_usuarios", type=int, default=10,
                         help="cuantos usuarios de test explicar con LIME")
    parser.add_argument("--n_muestras", type=int, default=200,
                         help="cuantas perturbaciones LIME por usuario")
    args = parser.parse_args()

    if args.dataset in ("olist", "all"):
        procesar("olist", epochs_ramas=args.epochs_ramas, epochs_fusion=args.epochs_fusion,
                  n_usuarios=args.n_usuarios, n_muestras=args.n_muestras)
    if args.dataset in ("amazon", "all"):
        procesar("amazon", epochs_ramas=args.epochs_ramas, epochs_fusion=args.epochs_fusion,
                  n_usuarios=args.n_usuarios, n_muestras=args.n_muestras)
