import cv2
import gui

cap = cv2.VideoCapture(0)

while cap.isOpened():
    cap.grab()
    _, image = cap.retrieve()

    gui.display(image)
    try:
        gui.refresh()
    except KeyboardInterrupt:
        break

gui.close()