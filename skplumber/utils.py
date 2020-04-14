import colorlog
import signal

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
)

logger = colorlog.getLogger("skplumber")
logger.addHandler(handler)
logger.setLevel("INFO")


class EvaluationTimeoutError(Exception):
    pass


class PipelineRunError(Exception):
    pass


class conditional_timeout:
    """
    Can be used to exit a function if it's taking too long. E.g.
    for a function `foo`, this can be done:

    ```
    # Will raise `EvaluationTimeoutError` if `foo` takes longer than 5 seconds.
    with conditional_timeout(5):
        foo()
    ```
    """

    def __init__(
        self, seconds=1, should_timeout: bool = True, error_message="Timeout"
    ) -> None:
        self.seconds = seconds
        self.should_timeout = should_timeout
        self.error_message = error_message

    def handle_timeout(self, signum, frame) -> None:
        raise EvaluationTimeoutError(self.error_message)

    def __enter__(self) -> None:
        if self.should_timeout:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback) -> None:
        if self.should_timeout:
            signal.alarm(0)
