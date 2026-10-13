"""
main.py - start LumiTrack.

Run from the project folder with:
    python main.py
"""
import sys


def check_dependencies():
    """Give a friendly message instead of a long traceback if a package is missing."""
    missing = []
    for module, package in (("cv2", "opencv-python"), ("numpy", "numpy"), ("PIL", "Pillow")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install them with:  pip install -r requirements.txt")
        sys.exit(1)


def main():
    check_dependencies()
    import tkinter as tk
    from lumitrack.app import LumiTrackApp

    root = tk.Tk()
    LumiTrackApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
