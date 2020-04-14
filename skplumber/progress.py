from time import time
import typing as t
from scipy.stats import genextreme as gev

from skplumber.consts import OptimizationDirection


class EVProgress:
    """
    A class for using Extreme Value Theory to build of model
    of how the maximum or minimum of a distribution of independently drawn
    samples is behaving. Useful for when you're trying to draw
    samples from a distribution to get the highest values you
    can (e.g. when trying to find a machine learning pipeline that
    performs very well). This can give you an idea of how much time
    it will take until you find a new maximum.

    First, instantiate: `progress = EVProgress()`. Next, right before you
    start making observations, call `progress.start()` to begin timing each
    observation. Pass each observation `x` into `progress` via
    `progress.observe(x)`. Once `progress.block_size` observations are
    observed, the return time, the amount of time `progress` predicts will
    elapse before a new extremum will be observed, can be queried from
    `progress` via `progress.return_time`. You will know when the return
    time can be queried via the `progress.can_report` flag.

    Note that especially early on, the report time is likely to be innacurate
    because `progress` doesn't have enough observations yet.
    """

    def __init__(self, block_size: int, extremum: OptimizationDirection) -> None:
        """
        Parameters
        ----------
        block_size : int
            The number of observations to take a sample maximum from.
        extremum : OptimizationDirection
            Whether to keep track of the
            maximum or minimum values of the observations and fit
            a GEV model to that.
        """
        self.block_size = block_size
        # This will be wiped out every `block_size` observations
        self.block: t.List[float] = []

        # The extrema observed from each block.
        self.extrema: t.List[float] = []
        # The extremum out of all the observed extrema
        self.extremum: t.Optional[float] = None
        # The GEV distribution fit to the extrema
        self.dist = None
        # The probability of finding an extremum more
        # extreme than `self.extremum`.
        self.p: t.Optional[float] = None
        # The name of the probability density function to use
        # to find `self.p`
        self.find_p = "sf" if extremum == OptimizationDirection.MAXIMIZE else "cdf"
        self.take_extremum = max if extremum == OptimizationDirection.MAXIMIZE else min
        # The estimated number of observations it will take to
        # find a new extremum.
        self.return_period: t.Optional[float] = None
        # The estimated number of seconds it will take to
        # find a new extremum.
        self.return_time: t.Optional[float] = None

        self.n_observations = 0
        # Total time take to collect all observations so far.
        self.total_t = 0.0
        # Used for tracking the time between observations
        self.t: t.Optional[float] = None
        # Average time taken to collect an observation.
        self.avg_t = 0.0

        # `True` once we sample our first extremum
        # and fit the distribution. The return time
        # cannot be reported until this is set to `True`.
        self.can_report = False

    def start(self) -> None:
        self.t = time()

    def observe(self, x: float) -> None:
        if self.t is None:
            raise ValueError(
                "`EVProgress.start` must first be called "
                "before observations can be made"
            )

        self.total_t += time() - self.t  # How long it took to get this observation
        self.n_observations += 1
        self.avg_t = self.total_t / self.n_observations

        self.block.append(x)
        if len(self.block) == self.block_size:
            self._process_block()

        # Start timing how long it will take to get the next observation.
        self.t = time()

    def _process_block(self) -> None:
        assert len(self.block) == self.block_size
        # We've finished a block. Take the extremum for it
        # as an observation for the GEV distribution and reset the block.
        self.extrema.append(self.take_extremum(self.block))
        self.extremum = self.take_extremum(self.extrema)
        self.block.clear()
        # Fit a generalized extreme value (GEV) distribution to our
        # extremum samples.
        self.dist = gev(*gev.fit(self.extrema))
        self.p = getattr(self.dist, self.find_p)(self.extremum)
        self.return_period = 1.0 / (self.block_size * self.p)  # type: ignore
        self.return_time = self.return_period * self.avg_t
        self.can_report = True
