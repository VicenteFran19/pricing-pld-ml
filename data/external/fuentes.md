# Fuentes de datos externos

Este archivo documenta explícitamente qué es real y qué es sintético en este
proyecto. Ningún dato individual de cliente proviene de información real de
BBVA ni de ninguna otra entidad — toda la base de clientes es generada
(`data/synthetic/clientes_pld_sintetico.csv`). Lo que sí es real son los
**parámetros agregados** usados para calibrar esa generación.

## 1. Costo de fondeo — BCRP

- **Serie:** Tasa de Referencia de la Política Monetaria (código `PD04722MM`)
- **Archivo:** `bcrp_tasa_referencia_2021_2026.csv`
- **Rango descargado:** enero 2021 – agosto 2026
- **Uso en el proyecto:** se usa el **promedio de los últimos 12 meses**
  (set-2025 a ago-2026 ≈ 4.25%) como costo de fondeo fijo. No se modela la
  serie completa como input dinámico del pricing — es una decisión de
  alcance para mantener el proyecto enfocado en la mecánica de pricing, no
  en un modelo de series de tiempo macroeconómicas.
- **Fuente:** https://estadisticas.bcrp.gob.pe/estadisticas/series/mensuales/tasas-de-interes

## 2. Benchmark de tasas de mercado (peers) — SBS

- **Reporte:** Tasas Activas Anuales por Tipo de Crédito y Empresa Bancaria
  (Moneda Nacional), categoría "Préstamos no Revolventes para Libre
  Disponibilidad"
- **Archivo:** `sbs_tcea_pld_peers.csv`
- **Fecha de reporte:** 15/09/2026
- **Advertencia de calidad de dato:** el reporte publica **TEA (Tasa
  Efectiva Anual), no TCEA**. La TCEA (que incluye comisiones y seguro)
  suele ser algo mayor. Para el MVP se usa TEA por disponibilidad directa
  del reporte; queda documentado como limitación conocida.
- **Advertencia adicional:** la columna "≤360 días" del reporte original
  tiene valores erráticos (ej. BCP 91.8%) por bajo volumen de operaciones
  en ese plazo en los últimos 30 días útiles — **no se usa esa columna**.
  Se usa exclusivamente la columna ">360 días", más estable y representativa
  de un PLD estándar (12-60 meses).
- **Valores usados (TEA, PLD >360 días):**

  | Banco | TEA |
  |---|---|
  | BBVA | 18.00% |
  | BCP | 20.84% |
  | Interbank | 17.77% |
  | Scotiabank | 17.93% |
  | **Promedio (usado como benchmark)** | **18.6%** |

- **Fuente:** https://www.sbs.gob.pe/app/pp/EstadisticasSAEEPortal/Paginas/TIActivaTipoCreditoEmpresa.aspx?tip=B

## 3. Distribución de ingreso del cliente — INEI

- **Indicador:** Ingreso promedio mensual proveniente del trabajo, Lima
  Metropolitana
- **Periodo:** trimestre móvil junio-julio-agosto 2026
- **Valor:** S/ 2,308.4 (hombres S/ 2,585.4 / mujeres S/ 1,976.3)
- **Uso en el proyecto:** se usa como la **media** de una distribución
  lognormal para generar el ingreso sintético de cada cliente. El INEI no
  publica la distribución completa (percentiles/desviación estándar) en
  sus reportes de prensa, por lo que el coeficiente de variación (0.60) es
  un **supuesto de calibración propio**, no un dato de INEI — queda
  documentado como tal en el código (`generar_data_sintetica.py`).
- **Nota de interpretación:** este promedio es de la población ocupada de
  Lima Metropolitana en general, no específico de clientes bancarizados con
  acceso a crédito de consumo (que suelen tener ingresos algo mayores al
  promedio poblacional). No se hizo ajuste por este sesgo en el MVP.
- **Fuente:** INEI, difundido en prensa el 15/09/2026 (El Comercio, Andina).

## 4. Referencia de morosidad del sistema — SBS

- **Indicador:** índice de morosidad del crédito al consumo, sistema
  financiero supervisado
- **Valor:** 4.8% (cierre marzo 2026)
- **Uso en el proyecto:** se usa únicamente como referencia para calibrar
  que la morosidad **agregada** de la base sintética generada se acerque a
  un rango realista (la base generada resulta en ~4.9%). No se usó como
  input directo de ningún modelo.

## Supuestos de negocio (no provienen de ninguna fuente pública)

- **LGD (Loss Given Default):** 65%. Supuesto fijo usado en el cálculo de
  margen esperado. En un contexto real, este valor lo definiría el área de
  Riesgos con base en recuperaciones históricas.
- **Rango de búsqueda de tasas candidatas del optimizador:** 14%-32%,
  limitado al rango efectivamente representado en la data de entrenamiento
  (ver comentario en `src/optimizador_pricing.py`).
