"""Transformación de variables asimétricas, compatible con scikit-learn.

Regla (fijada antes de ver resultados): para cada columna se mide la asimetría
(skewness) en entrenamiento. Si |asimetría| ≤ `umbral`, la columna no se toca. Si es
mayor:
- columna estrictamente positiva → se prueban log10 y Box-Cox (Box & Cox 1964) y se
  elige la que deja la asimetría más cerca de 0;
- columna con ceros o negativos → Yeo-Johnson (Yeo & Johnson 2000).
Las columnas con dos valores o menos (indicadoras) nunca se transforman. Los λ se
estiman por máxima verosimilitud en entrenamiento y se aplican tal cual a validación.
λ se limita a [−3, 3]: en variables casi siempre nulas con unos pocos valores altos, la
máxima verosimilitud da λ extremos que vuelven la transformación inestable fuera de
entrenamiento. El reporte indica cuándo se aplicó el límite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


def _yj(x: np.ndarray, lam: float) -> np.ndarray:
    return stats.yeojohnson(x, lmbda=lam)


LAMBDA_MAX = 3.0


class TransformadorAsimetria(BaseEstimator, TransformerMixin):
    def __init__(self, umbral: float = 1.0, lambda_max: float = LAMBDA_MAX):
        self.umbral = umbral
        self.lambda_max = lambda_max

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(float)
        self.columnas_ = list(X.columns)
        self.metodo_, self.lambda_, filas = {}, {}, []
        for c in self.columnas_:
            v = X[c].to_numpy()
            v = v[np.isfinite(v)]
            sk0 = float(stats.skew(v)) if v.size > 2 else 0.0
            metodo, lam, sk1 = "ninguna", None, sk0
            if np.unique(v).size > 2 and abs(sk0) > self.umbral:
                L = self.lambda_max
                if v.min() > 0:
                    sk_log = float(stats.skew(np.log10(v)))
                    lam_bc = float(np.clip(stats.boxcox(v)[1], -L, L))
                    sk_bc = float(stats.skew(stats.boxcox(v, lmbda=lam_bc)))
                    metodo, lam, sk1 = ("log10", None, sk_log) if abs(sk_log) <= abs(sk_bc) else ("box-cox", lam_bc, sk_bc)
                else:
                    lam = float(np.clip(stats.yeojohnson(v)[1], -L, L))
                    metodo, sk1 = "yeo-johnson", float(stats.skew(stats.yeojohnson(v, lmbda=lam)))
                limitado = lam is not None and abs(lam) >= L
            else:
                limitado = False
            self.metodo_[c], self.lambda_[c] = metodo, lam
            filas.append({"variable": c, "asimetria_antes": sk0, "curtosis_antes": float(stats.kurtosis(v)) if v.size > 3 else np.nan,
                          "transformacion": metodo, "lambda": lam, "lambda_limitado": limitado, "asimetria_despues": sk1})
        self.reporte_ = pd.DataFrame(filas).set_index("variable")
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columnas_).astype(float).copy()
        for c in self.columnas_:
            m, lam = self.metodo_[c], self.lambda_[c]
            v = X[c].to_numpy()
            ok = np.isfinite(v)
            if m == "log10":
                v[ok] = np.log10(np.clip(v[ok], 1e-12, None))
            elif m == "box-cox":
                v[ok] = stats.boxcox(np.clip(v[ok], 1e-12, None), lmbda=lam)
            elif m == "yeo-johnson":
                v[ok] = _yj(v[ok], lam)
            v[~np.isfinite(v)] = np.nan       # un valor infinito se trata como ausente (lo imputa el paso siguiente)
            X[c] = v
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columnas_, dtype=object)
