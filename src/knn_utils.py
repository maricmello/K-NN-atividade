import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def proporcoes(y_tr, y_te):
    """Proporção de cada classe no treino e no teste (usada para comparar as divisões)."""
    return pd.DataFrame({
        "treino": y_tr.value_counts(normalize=True),
        "teste": y_te.value_counts(normalize=True),
    })


def metricas(y_real, y_prev):
    """Accuracy, precision, recall e F1 (macro) para um conjunto de previsões."""
    return {
        "accuracy": accuracy_score(y_real, y_prev),
        "precision": precision_score(y_real, y_prev, average="macro"),
        "recall": recall_score(y_real, y_prev, average="macro"),
        "f1": f1_score(y_real, y_prev, average="macro"),
    }


def escalar(tipo, X_a, X_b):
    """Ajusta o escalonador só em X_a (treino) e aplica nos dois conjuntos."""
    if tipo == "nenhum":
        return X_a.values, X_b.values
    sc = {"minmax": MinMaxScaler, "standard": StandardScaler}[tipo]()
    return sc.fit_transform(X_a), sc.transform(X_b)


def treinar_avaliar(X_treino, X_teste, y_treino, y_teste, tipo="minmax", k=5, p=2, pesos="uniform"):
    """Treina um K-NN (com o escalonamento indicado) e devolve as métricas de treino e de teste."""
    A, B = escalar(tipo, X_treino, X_teste)
    modelo = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=p, weights=pesos)
    modelo.fit(A, y_treino)
    tr = metricas(y_treino, modelo.predict(A))
    te = metricas(y_teste, modelo.predict(B))
    return {"f1_treino": tr["f1"], "f1_teste": te["f1"], "gap": tr["f1"] - te["f1"],
            "acc_teste": te["accuracy"], "prec_teste": te["precision"], "rec_teste": te["recall"]}


def criar_rodar(X_train, X_test, y_train, y_test):
    """Devolve a função `rodar(...)` já amarrada ao treino/teste do notebook."""

    def rodar(tipo="minmax", k=5, p=2, pesos="uniform", X_a=None, X_b=None):
        """Treina um K-NN e devolve as métricas de treino e de teste."""
        X_a = X_train if X_a is None else X_a
        X_b = X_test if X_b is None else X_b
        return treinar_avaliar(X_a, X_b, y_train, y_test, tipo=tipo, k=k, p=p, pesos=pesos)

    return rodar


def f1_teste_com_vazamento(Escalonador, X, y):
    """F1 de teste do procedimento ERRADO: ajusta o escalonador em X inteiro antes de dividir.

    Serve só para mostrar o vazamento de dados (data leakage) por comparação.
    """
    Xs = Escalonador().fit_transform(X)          # fit em tudo, antes de dividir
    a, b, c, d = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
    m = KNeighborsClassifier(n_neighbors=5).fit(a, c)
    return f1_score(d, m.predict(b), average="macro")


def validacao_cruzada(X_treino, y_treino, tipo, k, p=2, pesos="uniform", cv=None):
    """F1 macro médio (e desvio) em validação cruzada estratificada."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    esc = MinMaxScaler() if tipo == "minmax" else StandardScaler()
    pipe = make_pipeline(esc, KNeighborsClassifier(n_neighbors=k, p=p, weights=pesos))
    s = cross_val_score(pipe, X_treino, y_treino, cv=cv, scoring="f1_macro")
    return s.mean(), s.std()


def criar_cv_f1(X_train, y_train, cv=None):
    """Devolve a função `cv_f1(...)` já amarrada ao treino do notebook."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def cv_f1(tipo, k, p=2, pesos="uniform"):
        return validacao_cruzada(X_train, y_train, tipo, k, p=p, pesos=pesos, cv=cv)

    return cv_f1


def razao_distancias(X_a, X_b):
    """Média de (distância ao vizinho mais próximo / distância média) para os pontos de teste."""
    A, B = escalar("minmax", X_a, X_b)
    D = pairwise_distances(B, A)
    return (D.min(axis=1) / D.mean(axis=1)).mean(), D.mean()
