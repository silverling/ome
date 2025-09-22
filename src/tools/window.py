import cv2


class Window:
    def __init__(self, name: str, ms_per_frame: float | int):
        self.name = name
        self.ms_per_frame = ms_per_frame
        cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

    def show(self, frame, elapsed_ms: float):
        cv2.imshow(self.name, frame)
        key = cv2.waitKey(max(1, int(self.ms_per_frame - elapsed_ms)))
        if key == ord(" "):  # space to pause
            while key := cv2.waitKey(0):
                if key == ord("q") or key == 27:  # 'q' or ESC to quit
                    return True
                if key == ord(" "):
                    break
        elif key == ord("q") or key == 27:
            return True

        return False
