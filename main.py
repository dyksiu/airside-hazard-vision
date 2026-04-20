import os

# Enable Hailo monitor before the app/runtime starts.
os.environ.setdefault("HAILO_MONITOR", "1")

from tkinter import Tk
from app import YoloVideoApp


if __name__ == "__main__":
    root = Tk()
    app = YoloVideoApp(root)
    root.mainloop()
