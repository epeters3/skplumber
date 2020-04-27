import typing as t
from collections import defaultdict
import copy

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
from skplumber.utils import logger, PipelineRunError
from skplumber.consts import OptimizationDirection
from skplumber.tuners.utils import TuneResult


def _range_rule(lbound, ubound) -> float:
    """
    Uses a modified version (dividing by 10 instead of 4) of
    the range rule heuristic to provide a very rough estimate
    of what a good standard deviation could be for a normal
    distribution to sample genetic mutations from for a
    hyperparameter having the range (lbound, ubound).
    """
    return (ubound - lbound) / 10


def _get_flexga_metas(pipeline: Pipeline, X: pd.DataFrame) -> t.Dict[str, ArgMeta]:
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
                    (pmeta.lbound, pmeta.ubound),
                    _range_rule(pmeta.lbound, pmeta.ubound),
                )
            elif isinstance(pmeta, FloatParamMeta):
                flexga_arg_meta = FloatArgMeta(
                    (pmeta.lbound, pmeta.ubound),
                    _range_rule(pmeta.lbound, pmeta.ubound),
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


def _get_params_from_flexga(flexga_params: dict) -> t.Dict[int, t.Dict[str, t.Any]]:
    """
    Converts flexga's flattened param dictionary to the nested
    dictionary `pipeline` uses.
    """
    params: t.Dict[int, t.Dict[str, t.Any]] = defaultdict(dict)
    for flexga_key, value in flexga_params.items():
        i, key = flexga_key.split(",")
        i = int(i)
        params[i][key] = value
    return params


def ga_tune(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    evaluator: t.Callable,
    metric: Metric,
    exit_on_pipeline_error: bool = True,
    **flexgakwargs,
) -> TuneResult:
    """
    Performs a genetic algorithm hyperparameter tuning on `pipeline`,
    returning the best score it could find and the number of evaluations
    it completed. Essentially performs a `.fit` operation on the pipeline,
    where the pipeine is fit with the best performing hyperparameter
    configuration it could find.

    Returns
    -------
    result : TuneResult
        A named tuple containing data about how the tuning process went.
    """
    # See what score the model gets without any tuning
    starting_params = pipeline.get_params()
    starting_score = evaluator(pipeline, X, y, metric)

    # keep track of how many iterations were completed
    n_evals = 1  # we already completed one

    def objective(*args, **flexga_params) -> float:
        """
        The objective function the genetic algorithm will
        try to maximize.
        """
        params = _get_params_from_flexga(flexga_params)
        nonlocal n_evals

        try:
            score = evaluator(pipeline, X, y, metric)
            pipeline.set_params(params)
        except PipelineRunError as e:
            logger.exception(e)
            if exit_on_pipeline_error:
                raise e
            # Pipelines that make errors are bad.
            # TODO: make this `None` or `np.nan` instead.
            score = metric.worst_value

        n_evals += 1
        # The genetic algorithm tries to maximize
        return -score if metric.opt_dir == OptimizationDirection.MINIMIZE else score

    # Use flexga to find the best hyperparameter configuration it can.
    optimal_score, _, optimal_flexga_params = flexga(
        objective, kwargsmeta=_get_flexga_metas(pipeline, X), **flexgakwargs
    )
    if metric.is_better_than(optimal_score, starting_score):
        optimal_params = _get_params_from_flexga(optimal_flexga_params)
        did_improve = True
    else:
        # The tuner couldn't find anything better than the params the
        # pipeline started with under the conditions given.
        optimal_score = starting_score
        optimal_params = starting_params
        did_improve = False

    pipeline.set_params(optimal_params)
    pipeline.fit(X, y)

    logger.info("tuning complete.")
    logger.info(f"found best pipeline configuration: {pipeline}")
    logger.info(f"found best validation score of {optimal_score}")
    return TuneResult(optimal_score, n_evals, did_improve)
