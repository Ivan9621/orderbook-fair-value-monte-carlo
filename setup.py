from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="orderbook-fair-value-mc",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Monte Carlo framework for order book fair value estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/orderbook-fair-value-monte-carlo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "orderbook-mc-demo=examples.demo:main",
        ],
    },
    keywords="order book, monte carlo, fair value, market microstructure, quantitative finance",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/orderbook-fair-value-monte-carlo/issues",
        "Source": "https://github.com/yourusername/orderbook-fair-value-monte-carlo",
        "Documentation": "https://github.com/yourusername/orderbook-fair-value-monte-carlo/blob/main/README.md",
    },
)
