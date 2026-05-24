import pandas as pd


def resumir_actividad_diaria(df, columna_fecha, columna_usuario, columna_evento):
    temp = df.copy()
    temp[columna_fecha] = pd.to_datetime(temp[columna_fecha]).dt.date

    resumen = temp.groupby(columna_fecha).agg(
        eventos=(columna_evento, "count"),
        usuarios_unicos=(columna_usuario, "nunique")
    ).reset_index()

    umbral = resumen["eventos"].quantile(0.75)
    resumen["dia_intenso"] = resumen["eventos"] > umbral

    return resumen


