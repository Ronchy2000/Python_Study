import gui
import cv2
from freenect import sync_get_video as get_video

while True:
    image, _ = get_video()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gui.display(image)
    try:
        gui.refresh()
    except KeyboardInterrupt:
        break

gui.close()
