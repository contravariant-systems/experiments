from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="contravariant",
    version="0.1.0",
    description="Automatic Mechanics from Variational Principles",
    packages=find_packages(),
    python_requires=">=3.13",
    install_requires=requirements,
)
