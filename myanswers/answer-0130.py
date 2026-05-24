import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso


def predecir_eficiencia_reducida(X, y, n_componentes):
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X)

    model = Lasso(alpha=0.5)
    model.fit(X_pca, y)

    var_acumulada = np.sum(pca.explained_variance_ratio_)

    return {
        'modelo': model,
        'varianza_total': float(var_acumulada),
        'coeficientes': model.coef_
    }