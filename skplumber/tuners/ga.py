import typing as t
from collections import defaultdict

import pandas as pd
from flexga import flexga
from flexga.utils import inverted
from flexga.argmeta import (
    ArgMeta,
    FloatArgMeta,
    IntArgMeta,
    BoolArgMeta,
    CategoricalArgMeta,
)

from skplumber.pipeline import Pipeline
from skplumber.metrics import Metric
from skplumber.primitives.parammeta import (
    IntParamMeta,
    FloatParamMeta,
    BoolParamMeta,
    CategoricalParamMeta,
)
from skplumber.utils import logger
from skplumber.consts import OptimizationDirection


def range_rule(lbound, ubound) -> float:
    """
    Uses a modified version (dividing by 10 instead of 4) of
    the range rule heuristic to provide a very rough estimate
    of what a good standard deviation could be for a normal
    distribution to sample genetic mutations from for a
    hyperparameter having the range (lbound, ubound).
    """
    return (ubound - lbound) / 10


def get_flexga_metas(pipeline: Pipeline, X: pd.DataFrame) -> t.Dict[str, ArgMeta]:
    """
    Converts meta information about the hyperparameters
    of a pipeline's primitive steps to the format the `flexga`
    package uses to know the bounds and characteristics of
    those hyperparameters (the things `flexga` is optimizing).
    """
    param_metas = pipeline.param_metas_with_data(X)
    kwargsmeta = {}

    for i, step_pmetas in param_metas.items():
        for key, pmeta in step_pmetas.items():
            flexga_key = f"{i},{key}"
            if isinstance(pmeta, IntParamMeta):
                flexga_arg_meta = IntArgMeta(
                    (pmeta.lbound, pmeta.ubound), range_rule(pmeta.lbound, pmeta.ubound)
                )
            elif isinstance(pmeta, FloatParamMeta):
                flexga_arg_meta = FloatArgMeta(
                    (pmeta.lbound, pmeta.ubound), range_rule(pmeta.lbound, pmeta.ubound)
                )
            elif isinstance(pmeta, BoolParamMeta):
                flexga_arg_meta = BoolArgMeta()
            elif isinstance(pmeta, CategoricalParamMeta):
                flexga_arg_meta = CategoricalArgMeta(pmeta.options)
            else:
                raise ValueError(
                    f"unsupported ParamMeta type {type(pmeta)} for {key} param"
                )
            kwargsmeta[flexga_key] = flexga_arg_meta

    return kwargsmeta


def set_params_from_flexga(pipeline: Pipeline, flexga_params: dict) -> None:
    """
    Converts flexga's flattened param dictionary to the nested
    dictionary `pipeline` uses, finally setting them on `pipeline`.
    """
    params: t.Dict[int, t.Dict[str, t.Any]] = defaultdict(dict)
    for flexga_key, value in flexga_params.items():
        i, key = flexga_key.split(",")
        i = int(i)
        params[i][key] = value
    pipeline.set_params(params)


def ga_tune(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    evaluator: t.Callable,
    metric: Metric,
    **flexgakwargs,
) -> float:
    """
    Performs a genetic algorithm hyperparameter tuning on `pipeline.`
    """

    def objective(*args, **flexga_params) -> float:
        """
        The objective function the genetic algorithm will
        try to maximize.
        """
        set_params_from_flexga(pipeline, flexga_params)
        score = evaluator(pipeline, X, y, metric)
        # The genetic algorithm tries to maximize
        return -score if metric.opt_dir == OptimizationDirection.MINIMIZE else score

    # Use flexga to find the best hyperparameter configuration it can.
    optimal_score, _, optimal_flexga_params = flexga(
        objective, kwargsmeta=get_flexga_metas(pipeline, X), **flexgakwargs
    )
    # Use that best found hyperparameter configuration.
    set_params_from_flexga(pipeline, optimal_flexga_params)
    # Fit the pipeline on all the data using that configuration
    pipeline.fit(X, y)
    logger.info(f"tuning complete for pipeline {pipeline}")
    logger.info(f"found best validation score of {optimal_score}")
    return optimal_score
