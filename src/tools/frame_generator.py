import numba
import numpy as np


class FrameGenerator:
    def __init__(
        self,
        width: int,
        height: int,
        /,
        *,
        duration_ms=33,
        decay_time_us=15000,
        contrast_on=1.3,
        contrast_off=None,
        tonemapping_max_ev_count=5,
        tonemapping_factor=None,
    ) -> None:
        """
        Initialize the frame generator with the given parameters.

        Args:
            width (int): Width of the frame.
            height (int): Height of the frame.
            duration_ms (int): Duration in milliseconds for which the frame is generated.
            decay_time_us (float): Decay time in microseconds after which integrated frame tends back to neutral gray. This needs to be adapted to the scene dynamics.
            contrast_on (float): Contrast associated to ON events.
            contrast_off (float, optional): Contrast associated to OFF events. If negative, the inverse of contrast_on is used.
            tonemapping_max_ev_count (int): Maximum event count to tonemap in 8-bit grayscale frame. This needs to be adapted to the scene dynamic range & sensor sensitivity.
            tonemapping_factor (float, optional): Factor for tonemapping. Defaults to None, calculated based on other parameters.
        """

        self.width = width
        self.height = height
        self.duration_ms = duration_ms
        self.decay_time_us = decay_time_us
        self.contrast_on = contrast_on
        self.contrast_off = contrast_off or 1 / self.contrast_on
        self.tonemapping_max_ev_count = tonemapping_max_ev_count
        self.tonemapping_factor = tonemapping_factor or np.exp(
            -self.tonemapping_max_ev_count * np.log(self.contrast_on)
        )
        self.log_contrast_off = np.log(self.contrast_off)
        self.log_contrast_on = np.log(self.contrast_on)

        # Initialize internal state per pixel
        self.log_i = np.zeros((height, width), dtype=np.float32)  # log intensity
        self.last_t = np.zeros((height, width), dtype=np.float32)  # last timestamp

    def generate(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Process events to generate a grayscale frame."""
        assert len(x) == len(y) == len(p) == len(t)
        frame, self.log_i, self.last_t = events_window_to_grayscale(
            x,
            y,
            p,
            t,
            self.log_i,
            self.last_t,
            self.width,
            self.height,
            self.decay_time_us,
            self.log_contrast_on,
            self.log_contrast_off,
            self.tonemapping_factor,
        )
        return frame


# TODO: Pick a good name for this function, and optimize it using cupy.
@numba.njit(cache=True, fastmath=True)
def events_window_to_grayscale(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    log_i: np.ndarray,
    last_t: np.ndarray,
    /,
    *,
    width: int,
    height: int,
    decay_time_us: float,
    log_contrast_on: float,
    log_contrast_off: float,
    tonemapping_factor: float,
):
    for i in range(len(x)):
        x_i = x[i]
        y_i = y[i]

        if x_i >= width or y_i >= height:
            continue

        p_i = p[i]
        t_i = t[i]

        decay_factor = np.exp(-(t_i - last_t[y_i, x_i]) / decay_time_us)

        """
            p_i == 0 -> log_contrast_off
            p_i == 1 -> log_contrast_on
        """
        log_i[y_i, x_i] = log_i[y_i, x_i] * decay_factor + (
            p_i * (log_contrast_on - log_contrast_off) + log_contrast_off
        )
        last_t[y_i, x_i] = t_i

    decay_factor = np.exp(-(t[-1] - last_t) / decay_time_us)
    last_t[...] = t[-1]
    log_i *= decay_factor
    frame = np.clip(255 * tonemapping_factor * np.exp(log_i), 0, 255).astype(np.uint8)

    return frame, log_i, last_t
