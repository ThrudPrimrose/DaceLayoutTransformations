from setuptools import setup, find_packages

setup(
    name="layout_and_schedule_transformations",
    version="0.1.0",
    description="A collection of layout and schedule transformations for DaCe.",
    author="Yakup Budanaz",
    packages=find_packages(),
    install_requires=[
        "dace",
        "numpy"
    ],
    python_requires=">=3.9",
)