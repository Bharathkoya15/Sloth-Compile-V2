"""
dashboard.py
Local Streamlit Web Dashboard for Sloth-Compile V2: Multimodal Drop-Box Edition.
Main Screen Side-by-Side Workspace:
1. Left Panel  : Paste a Question -> Get the Direct Answer, Explanation, Python Code & Live Output
2. Right Panel : Paste Python Code Directly -> Tweak Code In-Memory without creating .py files
3. Bottom Panel: Real-time Drop-Box History & Unified Diff Inspector
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import streamlit as st

from config import (
    DEFAULT_TEXT_PROMPT,
    DEFAULT_VISION_PROMPT,
    DROPZONE_DIR,
    PROCESSED_DIR,
    TEXT_MODEL,
    VISION_MODEL,
    ensure_directories,
)
from database import get_recent_fixes, init_db, insert_fix
from diff_engine import generate_unified_diff
from sloth_llm import (
    answer_question,
    call_text_model,
    call_vision_model_bytes,
)

st.set_page_config(
    page_title="Sloth-Compile V2 | Question Solver & Code Tweaker",
    page_icon="🦥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    """Apply sloth-themed custom styling to the Streamlit interface."""
    st.markdown(
        """
        <style>
        .sloth-banner {
            background: linear-gradient(135deg, #1b2a1f 0%, #2d4635 100%);
            padding: 1.4rem 2rem;
            border-radius: 12px;
            border: 1px solid #456b52;
            color: #eef5f0;
            margin-bottom: 1.25rem;
        }
        .sloth-banner h1 {
            margin: 0;
            font-size: 2.0rem;
            color: #d8f3dc;
        }
        .sloth-banner p {
            margin: 0.35rem 0 0 0;
            font-size: 0.98rem;
            color: #b7e4c7;
        }
        .answer-card {
            background-color: #16241a;
            border-left: 5px solid #52b788;
            padding: 1rem 1.25rem;
            border-radius: 8px;
            color: #eef5f0;
            margin-bottom: 1rem;
        }
        .badge-vision {
            background-color: #5a189a;
            color: #f3e5f5;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-text {
            background-color: #1d3557;
            color: #e0fbfc;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the Sloth-Compile V2 themed header banner."""
    st.markdown(
        f"""
        <div class="sloth-banner">
            <h1>🦥 Sloth-Compile V2: Question Solver & Direct Code Tweaker</h1>
            <p>
                <b>Left Box:</b> Paste any question to get the full answer + Python code + live output. &nbsp;|&nbsp;
                <b>Right Box:</b> Paste code directly to tweak it in-memory without creating a <code>.py</code> file.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_in_sandbox(code: str) -> tuple[bool, str, str]:
    """
    Verify Python code inside an isolated temporary sandbox with a 5-second timeout.
    Returns (passed, stdout, stderr).
    """
    if not code.strip():
        return True, "", ""

    with tempfile.TemporaryDirectory(prefix="sloth_ui_sandbox_") as temp_dir:
        sandbox_script = Path(temp_dir) / "in_memory_check.py"
        sandbox_script.write_text(code, encoding="utf-8")

        safe_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        try:
            result = subprocess.run(
                [sys.executable, "-I", str(sandbox_script)],
                cwd=temp_dir,
                env=safe_env,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", "TimeoutError: Script execution exceeded 5 seconds."
        except Exception as exc:
            return False, "", f"SandboxExecutionError: {exc}"


def format_sandbox_report(passed: bool, stdout_log: str, stderr_log: str) -> str:
    """Format sandbox stdout/stderr telemetry to append to the diff log."""
    report = (
        "\n\n# ===================================================================\n"
        f"# 🧪 SANDBOX EXECUTION VERIFICATION: {'PASSED (Exit Code 0)' if passed else 'FAILED'}\n"
        "# ===================================================================\n"
    )
    if stdout_log:
        report += "# [STDOUT]:\n# " + "\n# ".join(stdout_log.splitlines()) + "\n"
    if stderr_log:
        report += "# [STDERR]:\n# " + "\n# ".join(stderr_log.splitlines()) + "\n"
    return report


def main() -> None:
    ensure_directories()
    init_db()
    inject_custom_css()
    render_header()

    # Initialize session state so results stay visible on screen
    if "last_qa_result" not in st.session_state:
        st.session_state["last_qa_result"] = None
    if "last_tweak_result" not in st.session_state:
        st.session_state["last_tweak_result"] = None
    if "tweaker_prefill" not in st.session_state:
        st.session_state["tweaker_prefill"] = ""

    fixes = get_recent_fixes(limit=100)

    # Sidebar: Controls, File Drop-Box Uploader & History
    with st.sidebar:
        st.title("🦥 Sloth Control Center")

        col_btn, col_toggle = st.columns([1, 1])
        with col_btn:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col_toggle:
            auto_refresh = st.toggle(
                "Auto-Poll",
                value=False,
                help="Turn on to auto-refresh every 5s when dropping files into ./sloth_dropzone",
            )

        st.divider()
        st.subheader("📂 History Feed")

        if not fixes:
            st.info("No runs recorded yet. Paste a question or code on the main screen!")
            selected_fix = None
        else:
            options_map: dict[str, dict] = {}
            for item in fixes:
                is_vision = "Multimodal" in item["model_used"] or VISION_MODEL in item["model_used"]
                is_error = item["status"].startswith("ERROR") or "Failed" in item["status"]
                icon = "❌" if is_error else ("👁️" if is_vision else "📝")
                label = f"#{item['id']} | {icon} {item['filename']} ({item['timestamp'].split()[-1]})"
                options_map[label] = item

            selected_label = st.radio(
                "Select a past run to inspect:",
                options=list(options_map.keys()),
                index=0,
            )
            selected_fix = options_map[selected_label]

        st.divider()
        with st.expander("📥 Folder Dropzone Uploader", expanded=False):
            uploaded_files = st.file_uploader(
                "Drop .py and .png/.jpg files into folder",
                type=["py", "png", "jpg"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                saved_any = False
                for uploaded_file in uploaded_files:
                    target_path = Path(DROPZONE_DIR).resolve() / uploaded_file.name
                    if not target_path.exists():
                        target_path.write_bytes(uploaded_file.getvalue())
                        saved_any = True
                if saved_any:
                    st.success("Dropped into folder! Processing...")
                    time.sleep(1.0)
                    st.rerun()

    # =========================================================================
    # MAIN WORKSPACE: SIDE-BY-SIDE QUESTION SOLVER & DIRECT CODE TWEAKER
    # =========================================================================
    col_question, col_tweaker = st.columns(2, gap="large")

    # -------------------------------------------------------------------------
    # LEFT COLUMN: PASTE A QUESTION -> GET THE ANSWER & PYTHON CODE
    # -------------------------------------------------------------------------
    with col_question:
        st.subheader("💬 1. Paste a Question -> Get the Answer")
        st.caption(
            "Paste any question, coding problem, or homework prompt below. Sloth provides the direct answer, runnable Python code, and the computed output."
        )

        with st.form("question_solver_form", clear_on_submit=False):
            user_question = st.text_area(
                "Paste Your Question Here:",
                height=195,
                placeholder="Example: How do I check if a string is a palindrome in Python? Give me the explanation and the Python code.",
            )
            submit_question = st.form_submit_button(
                "💡 Get Answer & Python Code",
                type="primary",
                use_container_width=True,
            )

        if submit_question:
            if not user_question.strip():
                st.warning("Please paste a question inside the box above first.")
            else:
                with st.spinner(f"Sloth is solving your question with {TEXT_MODEL}..."):
                    try:
                        qa_data = answer_question(user_question)
                        answer_text = qa_data["answer"]
                        generated_code = qa_data["code"]

                        passed, stdout_log, stderr_log = run_in_sandbox(generated_code)
                        status_label = (
                            "ANSWERED & VERIFIED ✅" if passed else "ANSWERED (Sandbox Warning ⚠️)"
                        )

                        prompt_comment = (
                            "# QUESTION:\n"
                            + "\n".join(f"# {line}" for line in user_question.strip().splitlines())
                            + "\n\n# ANSWER SUMMARY:\n"
                            + "\n".join(f"# {line}" for line in answer_text.splitlines())
                            + "\n"
                        )
                        diff_str = generate_unified_diff(
                            original_code=prompt_comment,
                            fixed_code=generated_code or "# (Conceptual Answer - No Code Needed)\n",
                            filename="question_answer.py",
                        )
                        full_diff_log = diff_str + format_sandbox_report(passed, stdout_log, stderr_log)

                        short_q = user_question.strip().splitlines()[0][:26]
                        insert_fix(
                            filename=f"[Question] {short_q}...",
                            original_code=prompt_comment,
                            fixed_code=generated_code or answer_text,
                            diff_html=full_diff_log,
                            status=status_label,
                            model_used=f"{TEXT_MODEL} (Question Solver)",
                        )

                        st.session_state["last_qa_result"] = {
                            "question": user_question,
                            "answer": answer_text,
                            "code": generated_code,
                            "stdout": stdout_log,
                            "stderr": stderr_log,
                            "passed": passed,
                        }
                    except Exception as exc:
                        st.error(f"Error solving question: {exc}")

        # Display the latest Question Answer right below the box
        if st.session_state["last_qa_result"] is not None:
            qa_res = st.session_state["last_qa_result"]
            st.markdown("#### ✅ Direct Answer & Explanation")
            st.markdown(
                f"<div class='answer-card'>{qa_res['answer']}</div>",
                unsafe_allow_html=True,
            )

            if qa_res["code"].strip():
                st.markdown("#### 🐍 Python Code Solution")
                st.code(qa_res["code"], language="python", line_numbers=True)

                if st.button("➡️ Load This Code into Direct Tweaker (Right Box)", use_container_width=True):
                    st.session_state["tweaker_prefill"] = qa_res["code"]
                    st.rerun()

            if qa_res["stdout"] or qa_res["stderr"]:
                st.markdown("#### 🖥️ Computed Output (Sandbox Result)")
                st.code(
                    f"Output:\n{qa_res['stdout'] or '<No stdout printed>'}"
                    + (f"\n\nErrors:\n{qa_res['stderr']}" if qa_res["stderr"] else ""),
                    language="text",
                )

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: PASTE PYTHON CODE -> TWEAK DIRECTLY IN-MEMORY
    # -------------------------------------------------------------------------
    with col_tweaker:
        st.subheader("🛠️ 2. Paste Code -> Tweak Directly")
        st.caption(
            "Paste your Python code here to fix bugs or tweak it directly in the dashboard without converting it into a `.py` file."
        )

        with st.form("direct_code_tweaker_form", clear_on_submit=False):
            pasted_code = st.text_area(
                "Paste Python Code Here:",
                value=st.session_state["tweaker_prefill"],
                height=195,
                placeholder="def calculate_discount(price, percent):\n    return price - (price * percent / 0)\n\nprint(calculate_discount(100, 15))",
            )
            tweak_instruction = st.text_input(
                "How should Sloth tweak this code? (Leave blank to auto-fix all bugs):",
                placeholder="e.g., Fix the bug, handle negative numbers, and add docstrings.",
            )
            optional_screenshot = st.file_uploader(
                "Optional: Attach an Error Screenshot (.png / .jpg)",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=False,
            )
            submit_tweak = st.form_submit_button(
                "⚡ Tweak Code Directly (No File Needed)",
                type="primary",
                use_container_width=True,
            )

        if submit_tweak:
            if not pasted_code.strip():
                st.warning("Please paste some Python code inside the box above first.")
            else:
                with st.spinner("Sloth is tweaking and verifying your code in-memory..."):
                    try:
                        custom_prompt = (
                            f"{DEFAULT_TEXT_PROMPT}\n\nUSER TWEAK REQUEST:\n{tweak_instruction.strip()}"
                            if tweak_instruction.strip()
                            else DEFAULT_TEXT_PROMPT
                        )

                        if optional_screenshot is not None:
                            vis_prompt = (
                                f"{DEFAULT_VISION_PROMPT}\n\nUSER TWEAK REQUEST:\n{tweak_instruction.strip()}"
                                if tweak_instruction.strip()
                                else DEFAULT_VISION_PROMPT
                            )
                            tweaked_code = call_vision_model_bytes(
                                prompt=vis_prompt,
                                code=pasted_code,
                                image_bytes=optional_screenshot.getvalue(),
                            )
                            used_model = f"{VISION_MODEL} (In-Memory Multimodal)"
                        else:
                            tweaked_code = call_text_model(
                                prompt=custom_prompt,
                                code=pasted_code,
                            )
                            used_model = f"{TEXT_MODEL} (In-Memory Direct Tweak)"

                        passed, stdout_log, stderr_log = run_in_sandbox(tweaked_code)
                        status_label = (
                            "FIXED & VERIFIED ✅ (In-Memory)" if passed else "TWEAKED (Sandbox Warning ⚠️)"
                        )

                        diff_str = generate_unified_diff(
                            original_code=pasted_code,
                            fixed_code=tweaked_code,
                            filename="pasted_snippet.py",
                        )
                        full_diff_log = diff_str + format_sandbox_report(passed, stdout_log, stderr_log)

                        insert_fix(
                            filename="[Direct Tweak] snippet",
                            original_code=pasted_code,
                            fixed_code=tweaked_code,
                            diff_html=full_diff_log,
                            status=status_label,
                            model_used=used_model,
                        )

                        st.session_state["last_tweak_result"] = {
                            "tweaked_code": tweaked_code,
                            "diff_log": full_diff_log,
                            "stdout": stdout_log,
                            "stderr": stderr_log,
                            "status": status_label,
                        }
                    except Exception as exc:
                        st.error(f"Failed to tweak code: {exc}")

        # Display the latest Direct Tweak result right below the box
        if st.session_state["last_tweak_result"] is not None:
            tw_res = st.session_state["last_tweak_result"]
            st.success(f"Status: **{tw_res['status']}**")
            st.markdown("#### 🟢 Tweaked Python Code")
            st.code(tw_res["tweaked_code"], language="python", line_numbers=True)

            if tw_res["stdout"] or tw_res["stderr"]:
                st.markdown("#### 🖥️ Sandbox Execution Output")
                st.code(
                    f"Output:\n{tw_res['stdout'] or '<No stdout printed>'}"
                    + (f"\n\nErrors:\n{tw_res['stderr']}" if tw_res['stderr'] else ""),
                    language="text",
                )

            with st.expander("🔍 View Unified Diff of Changes", expanded=False):
                st.code(tw_res["diff_log"], language="diff", line_numbers=False)

    st.divider()

    # =========================================================================
    # BOTTOM PANEL: FULL HISTORY & DIFF INSPECTOR
    # =========================================================================
    with st.expander("📜 Full History & Diff Inspector (Past Questions, Tweaks & Dropped Files)", expanded=False):
        if selected_fix is None:
            st.info("No past runs recorded yet.")
        else:
            is_vision_run = (
                VISION_MODEL in selected_fix["model_used"] or "Multimodal" in selected_fix["model_used"]
            )
            badge_class = "badge-vision" if is_vision_run else "badge-text"
            modality_tag = "👁️ VISION + CODE MULTIMODAL" if is_vision_run else "📝 TEXT / STUDIO ENGINE"

            h1, h2 = st.columns([2, 1])
            with h1:
                st.markdown(
                    f"### 📄 `{selected_fix['filename']}` &nbsp; <span class='{badge_class}'>{modality_tag}</span>",
                    unsafe_allow_html=True,
                )
            with h2:
                st.markdown(f"**Status:** `{selected_fix['status']}`  \n**Time:** `{selected_fix['timestamp']}`")

            c_orig, c_fixed = st.columns(2)
            with c_orig:
                st.markdown("**🔴 Input / Question**")
                st.code(selected_fix["original_code"], language="python", line_numbers=True)
            with c_fixed:
                st.markdown("**🟢 Output / Fixed Code**")
                st.code(selected_fix["fixed_code"], language="python", line_numbers=True)

            st.markdown("**🔍 Unified Diff & Sandbox Log**")
            st.code(selected_fix["diff_html"], language="diff", line_numbers=False)

    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()