"""Grouped cross-validation. Everything learned from data (imputation, scaling, PCA,
head selection, the classifier and its C) lives in a Pipeline fit on training-fold
rows only; folds are drawn over problem_id."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def folds(groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped folds with a seeded assignment of problems to folds; empty with fewer than two problems."""
    unique = np.unique(groups)
    if len(unique) < 2:
        return []
    rng = np.random.default_rng(seed)
    shuffled = dict(zip(unique, rng.permutation(len(unique))))
    keys = np.array([shuffled[g] for g in groups])
    return list(GroupKFold(n_splits=min(n_splits, len(unique))).split(np.zeros(len(groups)), groups=keys))


class GroupedLogistic(BaseEstimator):
    """L2 logistic regression with C chosen by inner grouped CV on the training rows."""

    def __init__(self, inner_splits: int = 3, Cs: tuple[float, ...] = C_GRID, seed: int = 0, n_jobs: int | None = None):
        self.inner_splits, self.Cs, self.seed, self.n_jobs = inner_splits, Cs, seed, n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> GroupedLogistic:
        self.model_ = None
        if len(np.unique(y)) < 2:                     # a training fold without positives: no model
            return self
        inner = [(train, test) for train, test in folds(groups, self.inner_splits, self.seed)
                 if len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2]
        if len(inner) >= 2:
            self.model_ = LogisticRegressionCV(Cs=list(self.Cs), cv=inner, scoring="roc_auc", class_weight="balanced",
                                               max_iter=3000, n_jobs=self.n_jobs).fit(X, y)
        else:                                          # too few positive problems for inner CV
            self.model_ = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(X, y)
        return self

    def score_rows(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), np.nan) if self.model_ is None else self.model_.decision_function(X)


class GroupedRidge(BaseEstimator):
    """Ridge regression for the graded target with alpha chosen by inner grouped CV."""

    def __init__(self, inner_splits: int = 3, alphas: tuple[float, ...] = ALPHA_GRID, seed: int = 0):
        self.inner_splits, self.alphas, self.seed = inner_splits, alphas, seed

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> GroupedRidge:
        inner = folds(groups, self.inner_splits, self.seed)
        if len(inner) >= 2:
            self.model_ = RidgeCV(alphas=list(self.alphas), cv=inner).fit(X, y)
        else:                                          # too few problems for inner CV
            self.model_ = Ridge(alpha=float(np.median(self.alphas))).fit(X, y)
        return self

    def score_rows(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)


class TopHeads(BaseEstimator, TransformerMixin):
    """Keep all base columns plus every column of the ``k`` attention heads that score best on the training fold.

    ``head_of`` gives the head id of each extra column (all statistics and delays of one
    (layer, head) share an id). A head's score is its best column's |AUROC − 0.5| for a
    binary target, or |Spearman| with a graded target.
    """

    def __init__(self, n_base: int, head_of: np.ndarray, k: int):
        self.n_base, self.head_of, self.k = n_base, head_of, k

    def fit(self, X: np.ndarray, y: np.ndarray) -> TopHeads:
        from .metrics import Ranked, _rank, binary_metrics, graded_metrics
        extra = X[:, self.n_base:]
        binary = set(np.unique(y)) <= {0, 1}
        w = np.ones(len(y))
        y_rank = None if binary else _rank(y)
        strength = np.zeros(extra.shape[1])
        for j in range(extra.shape[1]):
            column = extra[:, j]
            column = np.where(np.isfinite(column), column, np.nanmedian(column) if np.isfinite(column).any() else 0.0)
            if binary:
                strength[j] = abs(binary_metrics(Ranked.build(column), y.astype(float), w)["auroc"] - 0.5)
            else:
                strength[j] = abs(graded_metrics(column, y, w, ranks=(_rank(column), y_rank))["spearman"])
        heads = np.unique(self.head_of)
        best = np.array([np.nan_to_num(strength[self.head_of == head]).max() for head in heads])
        chosen = heads[np.argsort(-best)[: self.k]]
        self.heads_ = chosen
        self.keep_ = np.concatenate([np.arange(self.n_base), self.n_base + np.nonzero(np.isin(self.head_of, chosen))[0]])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.keep_]


class CappedPCA(BaseEstimator, TransformerMixin):
    """Randomized PCA to ``n_components``, capped by the training fold's own rows and columns."""

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: Any = None) -> CappedPCA:
        k = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
        self.pca_ = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca_.transform(X)


def preprocess(pca: int | None = None) -> list[tuple[str, Any]]:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]
    if pca:
        steps.append(("pca", CappedPCA(pca)))
    return steps


def fit_pipeline(steps: list[tuple[str, Any]], estimator: Any, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[Pipeline, Any]:
    pipe = Pipeline([(name, clone(step)) for name, step in steps])
    Xt = pipe.fit_transform(X, y)
    model = clone(estimator).fit(Xt, y, groups)
    return pipe, model


def out_of_fold(X: np.ndarray, y: np.ndarray, groups: np.ndarray, estimator: Any, steps: list[tuple[str, Any]],
                splits: list[tuple[np.ndarray, np.ndarray]], fit_hook: Callable[[np.ndarray, np.ndarray], None] | None = None) -> np.ndarray:
    """Out-of-fold scores: each fold's preprocessing and model see only that fold's training rows."""
    scores = np.full(len(y), np.nan)
    for train, test in splits:
        if fit_hook is not None:
            fit_hook(train, test)
        if len(train) == 0:                            # no eligible training rows in this fold: scores stay NaN
            continue
        pipe, model = fit_pipeline(steps, estimator, X[train], y[train], groups[train])
        scores[test] = model.score_rows(pipe.transform(X[test]))
    return scores
