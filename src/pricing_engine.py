"""
Pricing engine: punto único de entrada del sistema.

Dado el perfil de un cliente, devuelve la tasa óptima, el margen esperado,
y una comparación contra la tasa fija promedio del mercado (peers SBS) y
contra una política ingenua de "tasa única para todos".
"""

from dataclasses import asdict, dataclass

from src.modelo_riesgo import cargar_modelo as cargar_modelo_riesgo
from src.modelo_elasticidad import cargar_modelo as cargar_modelo_elasticidad
from src.optimizador_pricing import optimizar_tasa, evaluar_tasa, COSTO_FONDEO

TASA_MERCADO_PROMEDIO = 0.186  # SBS, peers PLD >360 días, ver generar_data_sintetica.py


@dataclass
class PerfilCliente:
    score_riesgo: float   # 0 (mínimo riesgo) a 1 (máximo riesgo)
    ingreso_mensual: float
    monto_solicitado: float
    plazo_meses: int


class PricingEngine:
    def __init__(self):
        self._modelo_riesgo = cargar_modelo_riesgo()
        self._modelo_elasticidad = cargar_modelo_elasticidad()

    def cotizar(self, cliente: PerfilCliente) -> dict:
        resultado = optimizar_tasa(
            score_riesgo=cliente.score_riesgo,
            ingreso=cliente.ingreso_mensual,
            monto=cliente.monto_solicitado,
            plazo=cliente.plazo_meses,
            modelo_elasticidad=self._modelo_elasticidad,
            modelo_riesgo=self._modelo_riesgo,
        )

        # Benchmark: qué pasaría si se ofreciera la tasa fija promedio de mercado a este mismo cliente
        p_acept_mercado, p_def_mercado, margen_mercado = evaluar_tasa(
            TASA_MERCADO_PROMEDIO,
            cliente.score_riesgo, cliente.ingreso_mensual,
            cliente.monto_solicitado, cliente.plazo_meses,
            self._modelo_elasticidad, self._modelo_riesgo,
        )

        mejora_absoluta = resultado.margen_esperado - margen_mercado
        mejora_pct = (mejora_absoluta / abs(margen_mercado) * 100) if margen_mercado != 0 else None

        return {
            "tasa_optima": round(resultado.tasa_optima, 4),
            "margen_esperado_optimo": round(resultado.margen_esperado, 2),
            "prob_aceptacion_optima": round(resultado.prob_aceptacion, 4),
            "prob_default_optima": round(resultado.prob_default, 4),
            "benchmark_tasa_fija_mercado": {
                "tasa": TASA_MERCADO_PROMEDIO,
                "margen_esperado": round(margen_mercado, 2),
                "prob_aceptacion": round(p_acept_mercado, 4),
            },
            "mejora_vs_tasa_fija_mercado": {
                "absoluta_soles": round(mejora_absoluta, 2),
                "porcentual": round(mejora_pct, 1) if mejora_pct is not None else None,
            },
            "costo_fondeo_usado": COSTO_FONDEO,
        }


if __name__ == "__main__":
    engine = PricingEngine()
    cliente_ejemplo = PerfilCliente(
        score_riesgo=0.35,
        ingreso_mensual=3500,
        monto_solicitado=15000,
        plazo_meses=36,
    )
    resultado = engine.cotizar(cliente_ejemplo)
    import json
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
