"""
Evalúa el pricing engine sobre una muestra del portafolio sintético y
compara el margen esperado agregado contra una política de tasa fija
igual al promedio de mercado (peers SBS). Esta es la métrica de negocio
central del proyecto: cuánto margen adicional genera la personalización
de tasa por cliente frente a una tasa única para todos.
"""

from pathlib import Path

import pandas as pd

from src.pricing_engine import PricingEngine, PerfilCliente

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic" / "clientes_pld_sintetico.csv"


def evaluar(n_muestra: int = 500, seed: int = 42):
    df = pd.read_csv(DATA_PATH).sample(n=n_muestra, random_state=seed)
    engine = PricingEngine()

    resultados = []
    for _, fila in df.iterrows():
        cliente = PerfilCliente(
            score_riesgo=fila["score_riesgo"],
            ingreso_mensual=fila["ingreso_mensual"],
            monto_solicitado=fila["monto_solicitado"],
            plazo_meses=int(fila["plazo_meses"]),
        )
        r = engine.cotizar(cliente)
        resultados.append(r)

    df_res = pd.DataFrame(resultados)
    margen_optimo_total = df_res["margen_esperado_optimo"].sum()
    margen_mercado_total = sum(r["benchmark_tasa_fija_mercado"]["margen_esperado"] for r in resultados)
    mejora_total_pct = (margen_optimo_total - margen_mercado_total) / abs(margen_mercado_total) * 100

    print(f"Muestra evaluada: {n_muestra} clientes")
    print(f"Margen esperado total — motor de pricing:      S/ {margen_optimo_total:,.0f}")
    print(f"Margen esperado total — tasa fija de mercado:   S/ {margen_mercado_total:,.0f}")
    print(f"Mejora agregada: {mejora_total_pct:+.1f}%")

    return df_res, margen_optimo_total, margen_mercado_total, mejora_total_pct


if __name__ == "__main__":
    evaluar()
