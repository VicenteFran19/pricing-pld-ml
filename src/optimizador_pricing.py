"""
Optimizador de pricing.

Para un cliente dado, barre un rango de tasas candidatas y, para cada una,
usa el modelo de elasticidad para estimar P(aceptación) y el modelo de
riesgo para estimar P(default). Con eso calcula el margen esperado:

    margen_esperado(tasa) = P(aceptación | tasa) * monto *
                            [ tasa - costo_fondeo - P(default) * LGD ]

Y elige la tasa que maximiza esa expresión. LGD (Loss Given Default) se
fija como supuesto de negocio (severidad de pérdida en caso de default),
no proviene de ninguna fuente pública — se documenta explícitamente como
supuesto.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

LGD_SUPUESTO = 0.65  # severidad de pérdida asumida en caso de default (supuesto de negocio, no de fuente pública)
COSTO_FONDEO = 0.0425  # BCRP, ver generar_data_sintetica.py

# Rango de búsqueda de tasas candidatas: se limita deliberadamente al rango
# efectivamente observado en la data de entrenamiento (tasa_ofrecida en el
# histórico simulado va de 12% a ~32%). Buscar más allá de ese rango haría
# que el optimizador se apoye en extrapolaciones del modelo de elasticidad
# fuera de la densidad de datos que vio en entrenamiento — con árboles de
# decisión eso produce estimaciones planas o inestables, no confiables.
# Ampliar este rango en producción requeriría antes ampliar el histórico
# real de tasas ofrecidas (más experimentación comercial), no solo cambiar
# este número.
TASAS_CANDIDATAS = np.arange(0.14, 0.33, 0.01)  # 14% a 32%


@dataclass
class ResultadoPricing:
    tasa_optima: float
    margen_esperado: float
    prob_aceptacion: float
    prob_default: float
    curva: pd.DataFrame  # margen esperado para cada tasa candidata (para graficar)


def evaluar_tasa(tasa, score_riesgo, ingreso, monto, plazo, modelo_elasticidad, modelo_riesgo):
    """Calcula P(aceptación), P(default) y margen esperado para una tasa dada."""
    X_elasticidad = pd.DataFrame([{
        "tasa_ofrecida": tasa,
        "score_riesgo": score_riesgo,
        "ingreso_mensual": ingreso,
        "monto_solicitado": monto,
        "plazo_meses": plazo,
    }])
    p_aceptacion = modelo_elasticidad.predict_proba(X_elasticidad)[0, 1]

    X_riesgo = pd.DataFrame([{
        "ingreso_mensual": ingreso,
        "monto_solicitado": monto,
        "plazo_meses": plazo,
        "score_riesgo": score_riesgo,
    }])
    p_default = modelo_riesgo.predict_proba(X_riesgo)[0, 1]

    margen_unitario = tasa - COSTO_FONDEO - p_default * LGD_SUPUESTO
    margen_esperado = p_aceptacion * monto * margen_unitario

    return p_aceptacion, p_default, margen_esperado


def optimizar_tasa(score_riesgo, ingreso, monto, plazo, modelo_elasticidad, modelo_riesgo) -> ResultadoPricing:
    filas = []
    for tasa in TASAS_CANDIDATAS:
        p_acept, p_def, margen = evaluar_tasa(
            tasa, score_riesgo, ingreso, monto, plazo, modelo_elasticidad, modelo_riesgo
        )
        filas.append({
            "tasa": tasa,
            "prob_aceptacion": p_acept,
            "prob_default": p_def,
            "margen_esperado": margen,
        })

    curva = pd.DataFrame(filas)
    fila_optima = curva.loc[curva["margen_esperado"].idxmax()]

    return ResultadoPricing(
        tasa_optima=float(fila_optima["tasa"]),
        margen_esperado=float(fila_optima["margen_esperado"]),
        prob_aceptacion=float(fila_optima["prob_aceptacion"]),
        prob_default=float(fila_optima["prob_default"]),
        curva=curva,
    )
