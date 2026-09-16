"""
Modelo de elasticidad (take-up): estima la probabilidad de que un cliente
acepte el crédito dado la tasa ofrecida y su perfil. Esta es la pieza que
representa la curva de demanda del "volumen" mencionado en la entrevista:
a mayor tasa ofrecida (todo lo demás constante), menor probabilidad de
aceptación.

Se entrena sobre TODA la base (aceptaron y no aceptaron), porque a
diferencia del default, la aceptación siempre se observa.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ["tasa_ofrecida", "score_riesgo", "ingreso_mensual", "monto_solicitado", "plazo_meses"]
TARGET = "acepto"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic" / "clientes_pld_sintetico.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_elasticidad.joblib"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)


def entrenar():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment("modelo_elasticidad_pld")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Restricción de monotonicidad: P(aceptación) debe ser no-creciente en tasa_ofrecida.
    # Sin esta restricción, el árbol puede "aplanarse" o incluso invertirse en zonas de
    # baja densidad de datos (tasas muy altas, poco frecuentes en el histórico simulado),
    # lo que produce comportamientos económicamente absurdos en el optimizador
    # (ej. elegir una tasa altísima porque el modelo predice ahí, por error, una
    # aceptación igual o mayor que a tasas más bajas). El resto de features no tiene
    # una dirección de monotonicidad de negocio obligatoria, por lo que se deja libre (0).
    monotone_constraints = [-1 if f == "tasa_ofrecida" else 0 for f in FEATURES]

    with mlflow.start_run(run_name="lightgbm_elasticidad"):
        mlflow.log_params(PARAMS)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("monotone_constraints", monotone_constraints)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("tasa_aceptacion_base", float(y.mean()))

        modelo = lgb.LGBMClassifier(**PARAMS, verbose=-1, monotone_constraints=monotone_constraints)
        modelo.fit(X_train, y_train)

        pred_proba = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred_proba)
        mlflow.log_metric("auc_test", auc)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelo, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))

    print(f"Modelo de elasticidad entrenado. AUC (test): {auc:.4f}")
    print(f"Guardado en: {MODEL_PATH}")
    print(f"Run registrado en MLflow -> {MLFLOW_DB_PATH}")
    return modelo, auc


def cargar_modelo():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    entrenar()
