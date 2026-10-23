"""
main.py
CLI Entry Point for Sloth-Compile V2: Multimodal Drop-Box Edition.
Initializes directories and SQLite DB, starts the background folder watcher daemon thread,
and launches the local Streamlit dashboard on localhost:8501.
"""

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

from config import (
    DB_PATH,
    DROPZONE_DIR,
    OLLAMA_URL,
    PROCESSED_DIR,
    TEXT_MODEL,
    VISION_MODEL,
    ensure_directories,
)
from database import init_db
from folder_watcher import start_watcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [SLOTH-MAIN]    | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_startup_banner() -> None:
    """Display the Sloth-Compile V2 system architecture banner on startup."""
    banner = f"""
===========================================================================
   🦥 SLOTH-COMPILE V2: MULTIMODAL DROP-BOX EDITION
===========================================================================
   Dropzone Folder  : {Path(DROPZONE_DIR).resolve()}
   Processed Folder : {Path(PROCESSED_DIR).resolve()}
   SQLite Database  : {Path(DB_PATH).resolve()}
   Ollama Endpoint  : {OLLAMA_URL}
   Text Model       : {TEXT_MODEL}
   Vision Model     : {VISION_MODEL}
   Web Dashboard    : http://localhost:8501
===========================================================================
"""
    print(banner)


def launch_streamlit_dashboard() -> subprocess.Popen:
    """Spawn the Streamlit server subprocess on port 8501."""
    dashboard_script = str(Path(__file__).resolve().parent / "dashboard.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        dashboard_script,
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    logger.info("Launching Sloth Web Dashboard on http://localhost:8501 ...")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return process


def main() -> None:
    # 1. Initialize required directories and SQLite database
    ensure_directories()
    init_db()
    print_startup_banner()

    # 2. Spawn the folder watcher in a background daemon thread
    stop_event = threading.Event()
    watcher_thread = threading.Thread(
        target=start_watcher,
        args=(stop_event,),
        name="SlothFolderWatcherDaemon",
        daemon=True,
    )
    watcher_thread.start()

    # 3. Launch the Streamlit web dashboard via subprocess.Popen
    streamlit_proc = launch_streamlit_dashboard()

    # 4. Keep the main thread alive and handle graceful shutdown
    logger.info("System online. Drop `.py` files (and optional `.png`/`.jpg` errors) into '%s'.", DROPZONE_DIR)
    logger.info("Press Ctrl+C to stop Sloth-Compile V2.")
    try:
        while True:
            # Verify Streamlit subprocess hasn't unexpectedly exited
            if streamlit_proc.poll() is not None:
                logger.warning("Streamlit dashboard exited unexpectedly. Restarting...")
                streamlit_proc = launch_streamlit_dashboard()
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n")
        logger.info("KeyboardInterrupt received. Shutting down Sloth-Compile V2...")
    finally:
        stop_event.set()
        if streamlit_proc.poll() is None:
            logger.info("Terminating Streamlit server subprocess...")
            streamlit_proc.terminate()
            try:
                streamlit_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamlit_proc.kill()
        watcher_thread.join(timeout=3.0)
        logger.info("Shutdown complete. Stay slow, code fast! 🦥")


if __name__ == "__main__":
    main()