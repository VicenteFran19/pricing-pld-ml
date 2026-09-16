"""
Modelo de riesgo: estima la probabilidad de default (PD) de un cliente,
dado su perfil. Se entrena solo sobre clientes que aceptaron el crédito
(porque default solo es observable si el crédito se desembolsó).

En un escenario real de Pricing, este modelo normalmente viene dado por
el equipo de Riesgo (score de bureau, modelo interno de PD). Aquí se
entrena desde cero porque no existe ese insumo real disponible, pero el
diseño del pricing_engine ya está preparado para reemplazar esta pieza
por un score de riesgo externo sin tocar el resto del sistema.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ["ingreso_mensual", "monto_solicitado", "plazo_meses", "score_riesgo"]
TARGET = "default"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic" / "clientes_pld_sintetico.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_riesgo.joblib"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)


def entrenar():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("modelo_riesgo_pld")

    df = pd.read_csv(DATA_PATH)
    df_aceptados = df[df["acepto"] == 1].dropna(subset=[TARGET])

    X = df_aceptados[FEATURES]
    y = df_aceptados[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="lightgbm_riesgo"):
        mlflow.log_params(PARAMS)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("tasa_default_base", float(y.mean()))

        modelo = lgb.LGBMClassifier(**PARAMS, verbose=-1)
        modelo.fit(X_train, y_train)

        pred_proba = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred_proba)
        mlflow.log_metric("auc_test", auc)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelo, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))

    print(f"Modelo de riesgo entrenado. AUC (test): {auc:.4f}")
    print(f"Guardado en: {MODEL_PATH}")
    print(f"Run registrado en MLflow -> {MLFLOW_DB_PATH}")
    return modelo, auc


def cargar_modelo():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    entrenar()
