from collections import deque
from app.schemas import FailureSignalVector
from config import get_settings

settings = get_settings()


class EMAState:
    """Maintains a single streaming EMA value."""

    def __init__(self, alpha: float) -> None:
        self._alpha    = alpha
        self._value:   float | None = None
        self._previous: float | None = None

    def update(self, x: float) -> float:
        self._previous = self._value
        if self._value is None:
            self._value = x
        else:
            self._value = self._alpha * x + (1 - self._alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value if self._value is not None else 0.0

    @property
    def previous(self) -> float | None:
        return self._previous

    def velocity(self) -> float:
        """Rate of change: current EMA minus previous EMA."""
        if self._previous is None or self._value is None:
            return 0.0
        return round(self._value - self._previous, 4)


class SignalEvolutionTracker:
    """
    Tracks multiple failure metrics via independent streaming EMAs.
    All public metrics are O(1) — no window iteration required.
    """

    def __init__(
        self,
        window_size: int | None = None,
        decay_alpha: float | None = None,
    ) -> None:
        alpha = decay_alpha or settings.tracker_decay_alpha
        if not (0.0 < alpha <= 1.0):
            raise ValueError("decay_alpha must be in (0.0, 1.0]")

        self._alpha = alpha
        self._count = 0

        # One EMA per tracked metric
        self._entropy_ema     = EMAState(alpha)
        self._agreement_ema   = EMAState(alpha)
        self._disagreement_ema = EMAState(alpha)
        self._high_risk_ema   = EMAState(alpha)

        # Lightweight history for degradation_velocity()
        # Stores only the high-risk EMA value after each update
        max_history = window_size or settings.tracker_window_size
        self._risk_history: deque[float] = deque(maxlen=max_history)

    def record(self, signal: FailureSignalVector) -> None:
        """Update all EMAs with a single new signal."""
        self._count += 1
        self._entropy_ema.update(signal.entropy_score)
        self._agreement_ema.update(signal.agreement_score)
        self._disagreement_ema.update(1.0 if signal.ensemble_disagreement else 0.0)
        risk_val = self._high_risk_ema.update(1.0 if signal.high_failure_risk else 0.0)
        self._risk_history.append(risk_val)

    def current_window_size(self) -> int:
        return self._count

    # Public EMA metrics 

    def average_entropy(self) -> float:
        return round(self._entropy_ema.value, 4)

    def average_agreement(self) -> float:
        return round(self._agreement_ema.value, 4)

    def disagreement_rate(self) -> float:
        return round(self._disagreement_ema.value, 4)

    def high_risk_rate(self) -> float:
        """
        EMA of the high-risk binary flag.
        """
        return round(self._high_risk_ema.value, 4)

    # ── Velocity and degradation ─

    def degradation_velocity(self, history: list[float] | None = None) -> float:
        """
        Detects whether the rate of high-risk failures is accelerating.
        """
        source = history if history is not None else list(self._risk_history)
        if len(source) < 2:
            return 0.0
        midpoint    = len(source) // 2
        first_half  = source[:midpoint]
        second_half = source[midpoint:]
        first_mean  = sum(first_half)  / len(first_half)
        second_mean = sum(second_half) / len(second_half)
        return round(second_mean - first_mean, 4)

    def is_degrading(self) -> bool:
        return (
            self.degradation_velocity() > settings.tracker_degradation_velocity_threshold
            or self.high_risk_rate()    > settings.tracker_degradation_risk_threshold
        )

    def trend_summary(self) -> dict:
        return {
            "signals_recorded":      self._count,
            "decay_alpha":           self._alpha,
            "ema_entropy":           self.average_entropy(),
            "ema_agreement":         self.average_agreement(),
            "ema_disagreement_rate": self.disagreement_rate(),
            "ema_high_risk_rate":    self.high_risk_rate(),
            "degradation_velocity":  self.degradation_velocity(),
            "is_degrading":          self.is_degrading(),
        }


# Singleton — imported directly by routes and failure_agent
evolution_tracker = SignalEvolutionTracker()
