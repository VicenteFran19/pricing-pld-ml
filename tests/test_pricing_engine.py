import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pricing_engine import PricingEngine, PerfilCliente
from src.optimizador_pricing import TASAS_CANDIDATAS


engine = PricingEngine()


def test_tasa_optima_dentro_del_rango_de_busqueda():
    cliente = PerfilCliente(score_riesgo=0.4, ingreso_mensual=3000, monto_solicitado=12000, plazo_meses=24)
    resultado = engine.cotizar(cliente)
    assert TASAS_CANDIDATAS.min() <= resultado["tasa_optima"] <= TASAS_CANDIDATAS.max()


def test_probabilidades_en_rango_valido():
    cliente = PerfilCliente(score_riesgo=0.6, ingreso_mensual=2500, monto_solicitado=8000, plazo_meses=36)
    resultado = engine.cotizar(cliente)
    assert 0 <= resultado["prob_aceptacion_optima"] <= 1
    assert 0 <= resultado["prob_default_optima"] <= 1


def test_cliente_mayor_riesgo_tiene_mayor_probabilidad_default_estimada():
    """A igualdad de otras variables, más score de riesgo -> mayor PD estimada en la tasa óptima."""
    base = dict(ingreso_mensual=3000, monto_solicitado=10000, plazo_meses=24)
    bajo_riesgo = engine.cotizar(PerfilCliente(score_riesgo=0.1, **base))
    alto_riesgo = engine.cotizar(PerfilCliente(score_riesgo=0.9, **base))
    assert alto_riesgo["prob_default_optima"] > bajo_riesgo["prob_default_optima"]


def test_output_tiene_las_llaves_esperadas():
    cliente = PerfilCliente(score_riesgo=0.3, ingreso_mensual=4000, monto_solicitado=20000, plazo_meses=48)
    resultado = engine.cotizar(cliente)
    llaves_esperadas = {
        "tasa_optima", "margen_esperado_optimo", "prob_aceptacion_optima",
        "prob_default_optima", "benchmark_tasa_fija_mercado",
        "mejora_vs_tasa_fija_mercado", "costo_fondeo_usado",
    }
    assert llaves_esperadas.issubset(resultado.keys())
