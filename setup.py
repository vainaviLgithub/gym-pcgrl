from setuptools import setup, find_packages

setup(
    name="gym-pcgrl",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
        "matplotlib",
        "stable-baselines3"
    ],
)
