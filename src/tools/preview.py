from pathlib import Path

import tyro

from ome.io.sample.uniform import UniformSampler
from ome.repr.frame import events_to_grayscale_count_cuda
from ome.utils.timer import Timer


def main(
    file: Path,
    reader: str,
    ms_per_frame: int = 33,
    display: bool = False,
):
    if reader == "tumvie":
        from ome.io.reader.tumvie import TUMVIEReader

        event_reader = TUMVIEReader(file)
    elif reader == "m3ed":
        from ome.io.reader.m3ed import M3EDReader

        event_reader = M3EDReader(file)
    elif reader == "vector":
        from ome.io.reader.vector import VECTORReader

        event_reader = VECTORReader(file)
    elif reader == "dsec":
        from ome.io.reader.dsec import DSECReader

        event_reader = DSECReader(file)
    elif reader == "mvsec":
        from ome.io.reader.mvsec import MVSECReader

        event_reader = MVSECReader(file)
    else:
        raise ValueError(f"Unknown reader type: {reader}")

    if display:
        import cv2

        import tools.window

        window = tools.window.Window(file.name, ms_per_frame)

    sampler = UniformSampler(event_reader, sample_rate=1, duration_ms=ms_per_frame)
    timer = Timer()
    for idx, tick in enumerate(sampler.get_all_timestamps(anchor="start")):
        timer.tick()
        x, y, p, t = sampler[idx]
        frame = events_to_grayscale_count_cuda(x, y, width=event_reader.width, height=event_reader.height)
        elapsed = timer.elapsed()

        print(
            f"[{idx + 1}/{len(sampler)}] | Number of events: {len(x)}"
            f" | Processing time: {elapsed * 1e3:.2f} ms"
            f" | Event time: {tick} - {tick + ms_per_frame} ms"
        )

        if display and window.show(frame, elapsed * 1e3):
            break

    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        pass
