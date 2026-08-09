"""Create a native, self-contained Shadow Reader desktop bundle.

Run this file on the operating system and CPU architecture you want to ship.
The output contains Python, all Python packages, the web UI and FFmpeg.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_NAME = "ShadowReader"
DIST_DIR = ROOT / "dist"
RELEASE_DIR = ROOT / "release"
USER_GUIDE = RELEASE_DIR / "Shadow Reader 使用说明.docx"


def platform_label() -> str:
    system = platform.system().lower()
    aliases = {"darwin": "macos", "windows": "windows", "linux": "linux"}
    machine = platform.machine().lower()
    machine = {
        "amd64": "x64",
        "x86_64": "x64",
        "aarch64": "arm64",
    }.get(machine, machine)
    return f"{aliases.get(system, system)}-{machine}"


def build_user_guide() -> Path:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_user_guide.py")],
        cwd=ROOT,
        check=True,
    )
    if not USER_GUIDE.exists():
        raise RuntimeError(f"Expected user guide was not created: {USER_GUIDE}")
    return USER_GUIDE


def archive_bundle(bundle: Path, label: str, user_guide: Path) -> Path:
    RELEASE_DIR.mkdir(exist_ok=True)
    archive_base = RELEASE_DIR / f"{APP_NAME}-{label}"
    (ROOT / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"release-stage-{label}-",
        dir=ROOT / "build",
    ) as stage_dir:
        stage = Path(stage_dir)
        shutil.copytree(bundle, stage / bundle.name)
        shutil.copy2(user_guide, stage / user_guide.name)
        archive_format = "gztar" if platform.system() == "Linux" else "zip"
        return Path(shutil.make_archive(str(archive_base), archive_format, stage))


def sign_and_notarize_macos_bundle(bundle: Path) -> None:
    """Optionally make a macOS package pass Gatekeeper on first launch.

    Local builds remain usable without a certificate. A public distribution can
    set the two variables below from its release environment; credentials never
    live in this repository.
    """
    identity = os.environ.get("MACOS_SIGNING_IDENTITY")
    if not identity:
        print("macOS package is ad-hoc signed; a Developer ID is needed for Gatekeeper.")
        return

    subprocess.run(
        [
            "codesign", "--force", "--deep", "--options", "runtime",
            "--timestamp", "--sign", identity, str(bundle),
        ],
        check=True,
    )


def main() -> None:
    user_guide = build_user_guide()
    separator = ";" if platform.system() == "Windows" else ":"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name", APP_NAME,
        "--add-data", f"templates{separator}templates",
        "--collect-data", "imageio_ffmpeg",
        "--collect-all", "dashscope",
        "--collect-all", "edge_tts",
        "desktop.py",
    ]
    if platform.system() == "Darwin":
        command[command.index("--add-data"):command.index("--add-data")] = [
            "--osx-bundle-identifier", "com.shadowreader.desktop",
        ]
    subprocess.run(command, cwd=ROOT, check=True)

    bundle = DIST_DIR / (f"{APP_NAME}.app" if platform.system() == "Darwin" else APP_NAME)
    if not bundle.exists():
        raise RuntimeError(f"Expected bundle was not created: {bundle}")

    if platform.system() == "Darwin":
        sign_and_notarize_macos_bundle(bundle)
    archive = archive_bundle(bundle, platform_label(), user_guide)

    notary_profile = os.environ.get("MACOS_NOTARY_PROFILE")
    if platform.system() == "Darwin" and notary_profile:
        subprocess.run(
            ["xcrun", "notarytool", "submit", str(archive), "--keychain-profile", notary_profile, "--wait"],
            check=True,
        )
        subprocess.run(["xcrun", "stapler", "staple", str(bundle)], check=True)
        archive = archive_bundle(bundle, platform_label(), user_guide)

    print(f"Created: {archive}")


if __name__ == "__main__":
    main()
