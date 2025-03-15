"""
Package Setup
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if not line.startswith(("#", "-e", "-r"))
    ]

setup(
    name="legal_rag",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.10",
)
