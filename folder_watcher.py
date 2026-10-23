"""
folder_watcher.py
Watchdog daemon that monitors DROPZONE_DIR for dropped Python scripts and optional
error traceback screenshots, routes them to Ollama, verifies the fixed code inside
an isolated subprocess sandbox, self-heals on runtime failure, and logs to SQLite.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from watchdog.events import FileMovedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config import (
    DEFAULT_TEXT_PROMPT,
    DEFAULT_VISION_PROMPT,
    DROPZONE_DIR,
    IMAGE_PAIRING_WINDOW_SECONDS,
    PROCESSED_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    TEXT_MODEL,
    VISION_MODEL,
    WAIT_TIME_SECONDS,
    ensure_directories,
)
from database import insert_fix
from diff_engine import generate_unified_diff
from sloth_llm import call_text_model, call_vision_model

# Sandbox Execution Settings
SANDBOX_TIMEOUT_SECONDS: int = 5
MAX_SELF_HEAL_RETRIES: int = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [SLOTH-WATCHER] | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class SlothDropzoneHandler(FileSystemEventHandler):
    """
    Handles filesystem events inside DROPZONE_DIR.
    When a `.py` file arrives, waits 3 seconds to allow companion error screenshots
    (`.png` / `.jpg`) to land, dispatches the appropriate LLM agent, and verifies
    the generated fix inside an isolated subprocess sandbox.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._active_py_files: set[str] = set()
        self._consumed_images: set[str] = set()
        self._recent_image_drops: dict[str, float] = {}

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_path(str(event.src_path))

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            self._handle_path(str(event.dest_path))

    def _handle_path(self, raw_path: str) -> None:
        path = Path(raw_path).resolve()
        filename = path.name

        # Ignore hidden, temporary, or incomplete files
        if filename.startswith(".") or filename.endswith("~") or filename.endswith(".tmp"):
            return

        suffix = path.suffix.lower()

        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            with self._lock:
                self._recent_image_drops[str(path)] = time.time()
            logger.info("Detected dropped screenshot in dropzone: %s", filename)
            return

        if suffix == ".py":
            with self._lock:
                if str(path) in self._active_py_files:
                    return
                self._active_py_files.add(str(path))

            logger.info(
                "Detected dropped Python script: '%s'. Waiting %d seconds for optional screenshot...",
                filename,
                WAIT_TIME_SECONDS,
            )
            worker = threading.Thread(
                target=self._process_python_file,
                args=(path,),
                daemon=True,
            )
            worker.start()

    def _find_companion_image(self, py_path: Path, trigger_time: float) -> Path | None:
        """
        Look for a `.png` or `.jpg` file dropped in the same timeframe inside DROPZONE_DIR.
        """
        dropzone = Path(DROPZONE_DIR).resolve()
        candidates: list[tuple[float, Path]] = []
        now = time.time()

        with self._lock:
            for item in dropzone.iterdir():
                if not item.is_file() or item.name.startswith("."):
                    continue
                if item.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue

                resolved_str = str(item.resolve())
                if resolved_str in self._consumed_images:
                    continue

                recorded_drop_time = self._recent_image_drops.get(resolved_str)
                try:
                    stat_info = item.stat()
                    disk_time = max(stat_info.st_mtime, stat_info.st_ctime)
                except OSError:
                    continue

                effective_time = recorded_drop_time if recorded_drop_time is not None else disk_time

                if abs(effective_time - trigger_time) <= IMAGE_PAIRING_WINDOW_SECONDS or (
                    now - effective_time <= IMAGE_PAIRING_WINDOW_SECONDS
                ):
                    priority_score = effective_time + (1000.0 if item.stem == py_path.stem else 0.0)
                    candidates.append((priority_score, item.resolve()))

            if not candidates:
                return None

            candidates.sort(key=lambda entry: entry[0], reverse=True)
            chosen_image = candidates[0][1]
            self._consumed_images.add(str(chosen_image))
            return chosen_image

    @staticmethod
    def _verify_script_in_sandbox(code: str, filename: str) -> tuple[bool, str, str]:
        """
        Execute `code` inside an isolated temporary directory subprocess with a strict timeout.

        Returns:
            tuple[bool, str, str]: (passed_verification, stdout_output, stderr_output)
        """
        with tempfile.TemporaryDirectory(prefix="sloth_sandbox_") as temp_dir:
            sandbox_script = Path(temp_dir) / filename
            sandbox_script.write_text(code, encoding="utf-8")

            # Minimal safe environment variables required for Python to initialize on Windows/Linux
            safe_env = {
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "PATH": os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            try:
                # -I runs Python in isolated mode (ignores user site-packages & PYTHON* env vars)
                result = subprocess.run(
                    [sys.executable, "-I", str(sandbox_script)],
                    cwd=temp_dir,
                    env=safe_env,
                    capture_output=True,
                    text=True,
                    timeout=SANDBOX_TIMEOUT_SECONDS,
                )
                passed = result.returncode == 0
                return passed, result.stdout.strip(), result.stderr.strip()

            except subprocess.TimeoutExpired as exc:
                stdout_str = (exc.stdout or b"").decode("utf-8", errors="replace").strip() if isinstance(exc.stdout, bytes) else (exc.stdout or "").strip()
                timeout_msg = (
                    f"TimeoutError: Execution exceeded {SANDBOX_TIMEOUT_SECONDS} seconds "
                    "(possible infinite loop or blocking input() call)."
                )
                return False, stdout_str, timeout_msg
            except Exception as exc:
                return False, "", f"SandboxExecutionError: {exc}"

    def _process_python_file(self, py_path: Path) -> None:
        """
        Wait 3 seconds, route to Ollama, run sandbox verification, perform self-healing
        retry if needed, and log results to SQLite.
        """
        trigger_time = time.time()
        try:
            time.sleep(WAIT_TIME_SECONDS)

            if not py_path.exists():
                logger.warning("File '%s' disappeared before processing could begin.", py_path.name)
                return

            original_code = py_path.read_text(encoding="utf-8", errors="replace")
            if not original_code.strip():
                logger.warning("File '%s' is empty. Skipping LLM call.", py_path.name)
                insert_fix(
                    filename=py_path.name,
                    original_code="",
                    fixed_code="",
                    diff_html="# File was empty. Nothing to fix.",
                    status="SKIPPED (Empty File)",
                    model_used="N/A",
                )
                return

            companion_image = self._find_companion_image(py_path, trigger_time)

            # Pass 1: Initial LLM Fix (Vision or Text)
            if companion_image is not None and companion_image.exists():
                model_label = f"{VISION_MODEL} (Multimodal + {companion_image.name})"
                logger.info(
                    "Pairing '%s' with screenshot '%s' -> Routing to Vision Model (%s)...",
                    py_path.name,
                    companion_image.name,
                    VISION_MODEL,
                )
                fixed_code = call_vision_model(
                    prompt=DEFAULT_VISION_PROMPT,
                    code=original_code,
                    image_path=str(companion_image),
                )
                base_modality = f"Multimodal ({companion_image.name})"
            else:
                model_label = f"{TEXT_MODEL} (Text-Only)"
                logger.info(
                    "No screenshot paired with '%s' -> Routing to Text Model (%s)...",
                    py_path.name,
                    TEXT_MODEL,
                )
                fixed_code = call_text_model(
                    prompt=DEFAULT_TEXT_PROMPT,
                    code=original_code,
                )
                base_modality = "Text Analysis"

            # Sandbox Verification & Self-Healing Loop
            logger.info("Running sandbox verification for '%s'...", py_path.name)
            passed, stdout_log, stderr_log = self._verify_script_in_sandbox(fixed_code, py_path.name)

            attempt = 0
            while not passed and attempt < MAX_SELF_HEAL_RETRIES:
                attempt += 1
                logger.warning(
                    "Sandbox check failed on pass %d for '%s'. Triggering self-heal retry with stderr traceback...",
                    attempt,
                    py_path.name,
                )
                self_heal_prompt = (
                    f"{DEFAULT_TEXT_PROMPT}\n\n"
                    "CRITICAL SELF-HEALING CONTEXT:\n"
                    "Your previous attempt to fix this script still failed when executed in Python.\n"
                    "Here is the exact runtime traceback from the sandbox execution:\n"
                    f"---\n{stderr_log}\n---\n"
                    "Fix all remaining issues so the script executes cleanly with return code 0."
                )
                fixed_code = call_text_model(prompt=self_heal_prompt, code=fixed_code)
                model_label = f"{model_label} + Self-Heal ({TEXT_MODEL})"
                passed, stdout_log, stderr_log = self._verify_script_in_sandbox(fixed_code, py_path.name)

            if passed:
                if attempt > 0:
                    status_msg = f"FIXED & VERIFIED ✅ (Self-Healed on Retry #{attempt})"
                else:
                    status_msg = f"FIXED & VERIFIED ✅ ({base_modality})"
                logger.info("Sandbox verification PASSED for '%s'!Stdout: %s", py_path.name, stdout_log or "<empty>")
            else:
                status_msg = "FIXED (Verification Failed ⚠️)"
                logger.warning("Sandbox verification FAILED for '%s': %s", py_path.name, stderr_log)

            # Save the fixed Python file to PROCESSED_DIR/<filename>_fixed.py
            ensure_directories()
            stem = py_path.stem
            output_filename = f"{stem}_fixed.py"
            output_path = Path(PROCESSED_DIR).resolve() / output_filename
            output_path.write_text(fixed_code, encoding="utf-8")

            # Archive companion screenshot if present
            if companion_image is not None and companion_image.exists():
                archived_img_path = Path(PROCESSED_DIR).resolve() / f"{stem}_{companion_image.name}"
                try:
                    shutil.move(str(companion_image), str(archived_img_path))
                except OSError as move_err:
                    logger.debug("Could not move screenshot to processed folder: %s", move_err)

            # Generate unified diff and append sandbox execution telemetry
            diff_text = generate_unified_diff(
                original_code=original_code,
                fixed_code=fixed_code,
                filename=py_path.name,
            )

            sandbox_report = (
                "\n\n# ===================================================================\n"
                f"# 🧪 SANDBOX EXECUTION VERIFICATION: {'PASSED (Exit Code 0)' if passed else 'FAILED'}\n"
                "# ===================================================================\n"
            )
            if stdout_log:
                sandbox_report += f"# [STDOUT]:\n# " + "\n# ".join(stdout_log.splitlines()) + "\n"
            if stderr_log:
                sandbox_report += f"# [STDERR]:\n# " + "\n# ".join(stderr_log.splitlines()) + "\n"

            # Persist record to SQLite
            insert_fix(
                filename=py_path.name,
                original_code=original_code,
                fixed_code=fixed_code,
                diff_html=diff_text + sandbox_report,
                status=status_msg,
                model_used=model_label,
            )

            logger.info(
                "Completed processing '%s' -> Saved to '%s' [%s].",
                py_path.name,
                output_path,
                status_msg,
            )

        except Exception as exc:
            logger.error("Error processing '%s': %s", py_path.name, exc)
            fallback_original = ""
            if py_path.exists():
                try:
                    fallback_original = py_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    fallback_original = "# Unable to read source file."

            insert_fix(
                filename=py_path.name,
                original_code=fallback_original,
                fixed_code=f"# Automated fixing failed:\n# {exc}\n",
                diff_html=f"--- ERROR ---\n{exc}",
                status=f"ERROR: {type(exc).__name__}",
                model_used="Failed During Inference",
            )
        finally:
            with self._lock:
                self._active_py_files.discard(str(py_path))


def start_watcher(stop_event: threading.Event | None = None) -> Observer:
    """
    Start the Watchdog observer monitoring `DROPZONE_DIR`.
    """
    ensure_directories()
    event_handler = SlothDropzoneHandler()
    observer = Observer()
    observer.schedule(event_handler, path=os.path.abspath(DROPZONE_DIR), recursive=False)
    observer.start()
    logger.info("Sloth Drop-Box Watcher (with Sandbox Verifier) is monitoring: %s", os.path.abspath(DROPZONE_DIR))

    if stop_event is not None:
        try:
            while not stop_event.is_set():
                time.sleep(0.5)
        finally:
            observer.stop()
            observer.join()

    return observer