"""Expected empty-result signals for recoverable perception failures."""


class PerceptionEmptyError(RuntimeError):
    """A valid perception call produced no usable result."""
