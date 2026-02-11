"""
Setup script for backward compatibility.

Modern Python packaging uses pyproject.toml, but this file is provided
for compatibility with older tools and workflows.
"""

from setuptools import setup, find_packages

# Read version from package
with open("langgraph_rms/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A standalone Python library for integrating LangGraph-based multi-agent systems with Rules Management Systems"

setup(
    name="langgraph-rms-integration",
    version=version,
    description="A standalone Python library for integrating LangGraph-based multi-agent systems with Rules Management Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LangGraph RMS Team",
    python_requires=">=3.9",
    packages=find_packages(include=["langgraph_rms", "langgraph_rms.*"]),
    install_requires=[
        "pydantic>=2.0",
        "fastapi>=0.100",
        "httpx>=0.24",
        "langchain-core>=0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "hypothesis>=6.0",
            "respx>=0.20",
            "pytest-cov>=4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="langgraph rms rules multi-agent llm",
)
