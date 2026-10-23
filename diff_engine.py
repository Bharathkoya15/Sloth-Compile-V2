"""
diff_engine.py
Computes clean, unified diffs between original buggy code and LLM-repaired code.
"""

import difflib


def generate_unified_diff(
    original_code: str,
    fixed_code: str,
    filename: str = "script.py",
    context_lines: int = 3,
) -> str:
    """
    Compare `original_code` and `fixed_code` using `difflib.unified_diff`
    and return a clean diff string ready for Streamlit rendering.
    """
    original_lines = original_code.splitlines(keepends=True)
    fixed_lines = fixed_code.splitlines(keepends=True)

    # Ensure trailing newlines for clean diff formatting
    if original_lines and not original_lines[-1].endswith("\n"):
        original_lines[-1] += "\n"
    if fixed_lines and not fixed_lines[-1].endswith("\n"):
        fixed_lines[-1] += "\n"

    diff_generator = difflib.unified_diff(
        original_lines,
        fixed_lines,
        fromfile=f"a/{filename} (Original)",
        tofile=f"b/{Path_safe_stem(filename)}_fixed.py (Sloth Fixed)",
        n=context_lines,
    )

    diff_output = "".join(diff_generator)
    if not diff_output.strip():
        return "# No differences detected (the original code and fixed code are identical)."

    return diff_output


def Path_safe_stem(filename: str) -> str:
    """Extract filename stem without requiring external path objects."""
    if "." in filename:
        return filename.rsplit(".", 1)[0]
    return filename