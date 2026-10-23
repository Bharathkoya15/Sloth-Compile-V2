"""
config.py
Central configuration module for Sloth-Compile V2: Multimodal Drop-Box Edition.
"""

from pathlib import Path

# Core Directory & Database Paths
DROPZONE_DIR: str = "./sloth_dropzone"
PROCESSED_DIR: str = "./sloth_processed"
DB_PATH: str = "sloth_logs.db"

# Ollama Model & Server Configuration
TEXT_MODEL: str = "gemma4:e2b"
VISION_MODEL: str = "phi3:vision"
OLLAMA_URL: str = "http://localhost:11434"

# Operational Settings
WAIT_TIME_SECONDS: int = 3
IMAGE_PAIRING_WINDOW_SECONDS: float = 30.0
OLLAMA_TIMEOUT_SECONDS: int = 180
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")

# Default System Prompts
DEFAULT_TEXT_PROMPT: str = (
    "You are Sloth-Compile, an expert Python systems debugger and refactoring agent. "
    "Analyze the provided buggy Python code, identify all syntax, runtime, or logical defects, "
    "and fix them. Return ONLY the complete, valid, runnable fixed Python code. "
    "Do NOT wrap your output in markdown code fences (no ```python) and do NOT include any explanations."
)

DEFAULT_VISION_PROMPT: str = (
    "You are Sloth-Compile, an expert multimodal Python debugging agent. "
    "Inspect the attached screenshot containing the error traceback or terminal failure output, "
    "correlate it with the provided buggy Python script, and resolve the root cause of the error. "
    "Return ONLY the complete, valid, runnable fixed Python code. "
    "Do NOT wrap your output in markdown code fences (no ```python) and do NOT include any explanations."
)


def ensure_directories() -> None:
    """Ensure the dropzone and processed directories exist on disk."""
    Path(DROPZONE_DIR).mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)