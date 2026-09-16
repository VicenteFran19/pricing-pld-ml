"""
Generador de data sintética de clientes para Préstamo de Libre Disponibilidad (PLD).

IMPORTANTE: toda la data individual es 100% sintética. Los parámetros de
calibración (medias, rangos, tasas) provienen de fuentes públicas reales,
documentadas en data/external/fuentes.md. Ningún registro individual
proviene de datos reales de BBVA ni de ninguna entidad.

Fuentes de calibración:
- Costo de fondeo: BCRP, Tasa de Referencia de Política Monetaria (PD04722MM),
  promedio últimos 12 meses.
- Spread de mercado (peers PLD >360 días): SBS, Tasas Activas Anuales por
  Tipo de Crédito y Empresa Bancaria, consultado 15/09/2026.
- Ingreso del cliente: INEI, ingreso promedio mensual Lima Metropolitana,
  trimestre móvil jun-jul-ago 2026 (S/ 2,308.4).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Parámetros de calibración (documentados arriba, con fuente)
# ---------------------------------------------------------------------------

COSTO_FONDEO = 0.0425  # BCRP, promedio últimos 12m (set-25 a ago-26), ver bcrp_tasa_referencia_2021_2026.csv

# TEA promedio de mercado, PLD > 360 días (SBS, consultado 15/09/2026)
TASA_MERCADO_PEERS = {
    "BBVA": 0.1800,
    "BCP": 0.2084,
    "Interbank": 0.1777,
    "Scotiabank": 0.1793,
}
TASA_MERCADO_PROMEDIO = float(np.mean(list(TASA_MERCADO_PEERS.values())))  # ~18.6%

INGRESO_MEDIA = 2308.4  # INEI, Lima Metropolitana, trimestre jun-jul-ago 2026
INGRESO_CV = 0.60       # coeficiente de variación asumido (supuesto de calibración, no INEI)

MOROSIDAD_CONSUMO_SISTEMA = 0.048  # SBS, morosidad crédito consumo, cierre marzo 2026 (referencia de calibración del score de riesgo)


def _lognormal_params(media: float, cv: float) -> tuple[float, float]:
    """Convierte media y coef. de variación deseados a parámetros (mu, sigma) de una lognormal."""
    sigma2 = np.log(1 + cv ** 2)
    mu = np.log(media) - sigma2 / 2
    return mu, np.sqrt(sigma2)


def generar_clientes(n: int = 20_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Perfil del cliente ---
    mu, sigma = _lognormal_params(INGRESO_MEDIA, INGRESO_CV)
    ingreso_mensual = rng.lognormal(mean=mu, sigma=sigma, size=n)
    ingreso_mensual = np.clip(ingreso_mensual, 930, 25_000)  # piso ~RMV, techo razonable

    # Score de riesgo sintético (0 = mínimo riesgo, 1 = máximo riesgo).
    # Se simula correlacionado negativamente con el ingreso (a mayor ingreso, menor riesgo promedio),
    # con ruido idiosincrático — no es un modelo real de bureau, es un proxy para el ejercicio.
    ingreso_z = (np.log(ingreso_mensual) - np.log(ingreso_mensual).mean()) / np.log(ingreso_mensual).std()
    score_riesgo = 1 / (1 + np.exp(1.1 * ingreso_z + rng.normal(0, 1, n)))  # logística invertida
    score_riesgo = np.clip(score_riesgo, 0.01, 0.99)

    # Monto solicitado: entre 1 y 12 veces el ingreso mensual, con dispersión
    factor_monto = rng.gamma(shape=3.0, scale=1.7, size=n)
    monto_solicitado = np.clip(ingreso_mensual * factor_monto, 2_000, 80_000)
    monto_solicitado = np.round(monto_solicitado, -2)  # redondeado a centenas

    # Plazo en meses: típico de PLD, 12 a 60 meses
    plazo_meses = rng.choice([12, 18, 24, 36, 48, 60], size=n, p=[0.10, 0.15, 0.25, 0.25, 0.15, 0.10])

    # --- Experimento de pricing: tasa ofrecida ---
    # Para poder entrenar el modelo de elasticidad necesitamos variación de tasa
    # ofrecida a clientes similares (no una sola tasa fija por cliente).
    # Se simula un "spread ofrecido" alrededor del promedio de mercado, con
    # variación por score de riesgo (mayor riesgo -> mayor tasa ofrecida en promedio)
    # y ruido aleatorio (representa variación comercial/campañas).
    spread_riesgo = 0.030 + 0.20 * score_riesgo  # de 3 a 23 pp adicionales según riesgo
    ruido_comercial = rng.normal(0, 0.02, n)
    tasa_ofrecida = COSTO_FONDEO + spread_riesgo + ruido_comercial
    tasa_ofrecida = np.clip(tasa_ofrecida, 0.12, 0.55)

    # --- Variable objetivo 1: aceptación (elasticidad / take-up) ---
    # A mayor tasa relativa al mercado, menor probabilidad de aceptar.
    # A mayor score de riesgo (peor perfil), también menor probabilidad de aceptar
    # (clientes de mayor riesgo suelen tener menos alternativas pero también
    # menor capacidad de pago percibida -> se modela con un efecto moderado).
    delta_vs_mercado = tasa_ofrecida - TASA_MERCADO_PROMEDIO
    logit_aceptacion = 1.2 - 9.0 * delta_vs_mercado - 1.5 * score_riesgo + rng.normal(0, 0.4, n)
    prob_aceptacion = 1 / (1 + np.exp(-logit_aceptacion))
    acepto = rng.binomial(1, prob_aceptacion)

    # --- Variable objetivo 2: default (riesgo, solo observable si acepto=1) ---
    # Calibrado para que la morosidad agregada del segmento se acerque al
    # promedio del sistema (SBS, ~4.8%), pero variando fuerte con el score.
    logit_default = -5.6 + 4.5 * score_riesgo + rng.normal(0, 0.5, n)
    prob_default = 1 / (1 + np.exp(-logit_default))
    default = rng.binomial(1, prob_default)
    default = np.where(acepto == 1, default, np.nan)  # solo tiene sentido si el crédito se desembolsó

    df = pd.DataFrame({
        "cliente_id": np.arange(1, n + 1),
        "ingreso_mensual": np.round(ingreso_mensual, 2),
        "score_riesgo": np.round(score_riesgo, 4),
        "monto_solicitado": monto_solicitado,
        "plazo_meses": plazo_meses,
        "tasa_ofrecida": np.round(tasa_ofrecida, 4),
        "tasa_mercado_referencia": TASA_MERCADO_PROMEDIO,
        "costo_fondeo": COSTO_FONDEO,
        "acepto": acepto,
        "default": default,
    })
    return df


if __name__ == "__main__":
    df = generar_clientes()
    out_path = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "clientes_pld_sintetico.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Generados {len(df):,} clientes sintéticos -> {out_path}")
    print(f"Tasa de aceptación observada: {df['acepto'].mean():.1%}")
    print(f"Morosidad observada (entre aceptados): {df.loc[df['acepto']==1, 'default'].mean():.1%}")
    print(f"Tasa ofrecida promedio: {df['tasa_ofrecida'].mean():.1%} | mercado ref.: {TASA_MERCADO_PROMEDIO:.1%}")
