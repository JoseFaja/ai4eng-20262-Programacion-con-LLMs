import numpy as np
from sklearn.linear_model import LinearRegression


def regresion_log_segura(df, columna_x, columna_y):
    df_v = df[df[columna_y] > 0].copy()
    model = LinearRegression().fit(df_v[[columna_x]], np.log(df_v[columna_y]))
    return (model.intercept_, model.coef_[0])