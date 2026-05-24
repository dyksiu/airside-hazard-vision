import os
import sys
from tkinter import Tk
from app import YoloVideoApp


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    root = Tk()

    icon_path = resource_path("favicon.ico")
    if os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception:
            pass

    app = YoloVideoApp(root)
    root.mainloop()