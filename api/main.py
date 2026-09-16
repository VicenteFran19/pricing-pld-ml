"""
API de demostración del motor de pricing.

Ejecutar con:
    uvicorn api.main:app --reload

Luego abrir http://127.0.0.1:8000/docs para probar interactivamente.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.pricing_engine import PricingEngine, PerfilCliente

app = FastAPI(
    title="Motor de Pricing — Préstamo de Libre Disponibilidad (demo)",
    description=(
        "API de demostración. Toda la data usada para entrenar los modelos es sintética; "
        "los parámetros de calibración provienen de fuentes públicas (BCRP, SBS, INEI), "
        "documentadas en data/external/fuentes.md."
    ),
    version="0.1.0",
)

_engine: PricingEngine | None = None


def get_engine() -> PricingEngine:
    global _engine
    if _engine is None:
        _engine = PricingEngine()
    return _engine


class ClienteRequest(BaseModel):
    score_riesgo: float = Field(..., ge=0, le=1, description="0 = riesgo mínimo, 1 = riesgo máximo")
    ingreso_mensual: float = Field(..., gt=0, description="Ingreso mensual en soles")
    monto_solicitado: float = Field(..., gt=0, description="Monto del préstamo solicitado en soles")
    plazo_meses: int = Field(..., gt=0, le=60, description="Plazo del préstamo en meses")


@app.get("/")
def raiz():
    return {"status": "ok", "mensaje": "Motor de pricing PLD — demo. Ver /docs para probarlo."}


@app.post("/cotizar")
def cotizar(cliente: ClienteRequest):
    engine = get_engine()
    perfil = PerfilCliente(
        score_riesgo=cliente.score_riesgo,
        ingreso_mensual=cliente.ingreso_mensual,
        monto_solicitado=cliente.monto_solicitado,
        plazo_meses=cliente.plazo_meses,
    )
    return engine.cotizar(perfil)
