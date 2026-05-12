from __future__ import annotations

from pathlib import Path
import platform
import subprocess
from live_report_server import create_app


def main():
    base_dir = Path(__file__).resolve().parent.parent

    app = create_app(base_dir=base_dir)
    url = "http://127.0.0.1:5000/"

    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        elif system == "Windows":
            subprocess.run(["cmd", "/c", "start", "", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            subprocess.run(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass

    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
