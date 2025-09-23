from pathlib import Path

import tyro

from ome.repr.frame import events_to_grayscale_count_cuda
from ome.utils.timer import Timer


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
        from ome.io.reader.tumvie import TUMVIEReader

        event_reader = TUMVIEReader(file)
    elif reader == "m3ed":
        from ome.io.reader.m3ed import M3EDReader

        event_reader = M3EDReader(file)
    else:
        raise ValueError(f"Unknown reader type: {reader}")

    if display:
        import cv2

        import tools.window

        ms_per_frame_float = ms_per_frame if fps is None else 1000 / fps
        window = tools.window.Window(file.name, ms_per_frame_float)

    ticks = list(range(0, event_reader.max_ms, ms_per_frame))

    timer = Timer()
    for idx, tick in enumerate(ticks, start=1):
        timer.tick()
        x, y, p, t = event_reader.duration(tick, ms_per_frame)
        frame = events_to_grayscale_count_cuda(x, y, width=width, height=height)
        elapsed = timer.elapsed()

        print(f"[{idx}/{len(ticks)}] | Number of events: {len(x)} | Elapsed time: {elapsed * 1e3:.2f} ms")

        if display and window.show(frame, elapsed * 1e3):
            break

    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        pass
