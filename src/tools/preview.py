from pathlib import Path

import tyro

from ome.io.reader.tumvie import TUMVIEReader
from tools.frame_generator import events_to_grayscale_count_cuda
from tools.timer import Timer


def main(
    file: Path,
    width: int,
    height: int,
    reader: str,
    ms_per_frame: int = 33,
    fps: float | None = None,
    display: bool = False,
):
    if reader == "tumvie":
        event_reader = TUMVIEReader(file)
    else:
        raise ValueError(f"Unknown reader type: {reader}")

    if display:
        import cv2

        import tools.window

        ms_per_frame_float = ms_per_frame if fps is None else 1000 / fps
        window = tools.window.Window(file.name, ms_per_frame_float)

    max_events = 0
    timer = Timer()
    ticks = list(range(0, event_reader.max_ms, ms_per_frame))
    for idx, tick in enumerate(ticks, start=1):
        x, y, p, t = event_reader.duration(tick, ms_per_frame)

        max_events = max(max_events, len(x))
        timer.tick()
        frame = events_to_grayscale_count_cuda(x, y)
        elapsed = timer.elapsed()

        print(f"[{idx}/{len(ticks)}] Elapsed time: {elapsed * 1e3:.2f} ms")

        if display and window.show(frame, elapsed * 1e3):
            break

    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        pass
