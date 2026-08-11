#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  setup.sh — one-command setup for Linux / macOS
# ══════════════════════════════════════════════════════════════════
#  WHAT THIS DOES (in plain words):
#    1. Creates a private "virtual environment" (.venv) so the
#       project's libraries don't mess with the rest of your computer.
#    2. Installs the lightweight CPU-only version of "torch" first.
#       (The normal version bundles ~4 GB of NVIDIA GPU libraries that
#       are useless on computers without an NVIDIA GPU — like most
#       laptops. Installing the CPU build first means pip never
#       downloads all that junk.)
#    3. Installs every library listed in requirements.txt.
#    4. Removes one leftover NVIDIA package that xgboost drags in on
#       Linux (nvidia-nccl-cu13) — not needed on CPU-only machines.
#
#  HOW TO USE:
#    Open a terminal in this folder and run:   bash setup.sh
#
#  On Windows: see README.md (use the manual commands there instead).
# ══════════════════════════════════════════════════════════════════
set -e  # stop immediately if any step fails

# Go to the folder where this script lives (so it works from anywhere)
cd "$(dirname "$0")"

echo "▶ Step 1/4 — Creating virtual environment (.venv) ..."
python3 -m venv .venv

echo "▶ Step 2/4 — Upgrading pip (Python's package installer) ..."
./.venv/bin/python -m pip install --upgrade pip setuptools wheel

echo "▶ Step 3/4 — Installing CPU-only torch (no NVIDIA GPU needed) ..."
./.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "▶ Step 4/4 — Installing the rest of the project libraries ..."
./.venv/bin/python -m pip install -r requirements.txt

# On Linux, xgboost pulls in one useless NVIDIA package. Remove it.
echo "▶ Cleanup — removing unneeded NVIDIA package (CPU machine) ..."
./.venv/bin/python -m pip uninstall -y nvidia-nccl-cu13 2>/dev/null || true

echo
echo "✅ All done!"
echo
echo "Next steps:"
echo "  1. Put your API keys in a file named  .env   (see README.md)"
echo "  2. Start the app:   ./.venv/bin/streamlit run app/streamlit_app.py"
echo
