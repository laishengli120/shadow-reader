"""Desktop launcher for the self-contained Shadow Reader package."""

from __future__ import annotations

import atexit
import os
import platform
import subprocess
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from werkzeug.serving import BaseWSGIServer, make_server

from app import app


APP_NAME = "Shadow Reader"


@dataclass
class LocalServer:
    """A local-only Flask server with an automatically selected free port."""

    server: BaseWSGIServer
    thread: threading.Thread

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=3)


def start_server() -> LocalServer:
    # Binding to port 0 avoids collisions with other locally running apps.
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(
        target=server.serve_forever,
        name="shadow-reader-server",
        daemon=True,
    )
    thread.start()
    launcher = LocalServer(server=server, thread=thread)
    atexit.register(launcher.stop)
    return launcher


def open_in_browser(url: str) -> None:
    webbrowser.open_new_tab(url)


def run_desktop() -> None:
    launcher = start_server()

    # Useful for managed deployments and automated package smoke tests. Normal
    # desktop launches leave both variables unset and open the browser as usual.
    url_file = os.environ.get("SHADOW_READER_URL_FILE")
    if url_file:
        Path(url_file).write_text(launcher.url, encoding="utf-8")
    if os.environ.get("SHADOW_READER_NO_BROWSER") == "1":
        try:
            launcher.thread.join()
        except KeyboardInterrupt:
            pass
        return

    try:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
    except Exception:
        # The graphical launcher is unavailable only in headless environments.
        open_in_browser(launcher.url)
        if platform.system() == "Darwin":
            # Homebrew's Python intentionally omits Tk on many Macs. Use the
            # native dialog so the packaged app still has a visible Quit button.
            script = (
                'display dialog "Shadow Reader 正在浏览器中运行。关闭此窗口即可退出。" '
                'buttons {"退出"} default button "退出" with title "Shadow Reader"'
            )
            subprocess.run(["osascript", "-e", script], check=False)
            return

        # The server remains usable in a headless Linux environment as well.
        print(f"{APP_NAME} is running at {launcher.url}")
        try:
            launcher.thread.join()
        except KeyboardInterrupt:
            pass
        return

    root.title(APP_NAME)
    root.resizable(False, False)
    root.configure(padx=28, pady=24)

    ttk.Label(root, text=APP_NAME, font=("Arial", 18, "bold")).pack(pady=(0, 8))
    ttk.Label(root, text="已准备就绪，浏览器会自动打开。", font=("Arial", 11)).pack()
    ttk.Label(root, text="关闭此窗口即可退出应用。", font=("Arial", 9)).pack(pady=(4, 18))

    ttk.Button(
        root,
        text="打开 Shadow Reader",
        command=lambda: open_in_browser(launcher.url),
    ).pack(fill="x")

    def close() -> None:
        launcher.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.after(250, lambda: open_in_browser(launcher.url))
    root.mainloop()


if __name__ == "__main__":
    run_desktop()
