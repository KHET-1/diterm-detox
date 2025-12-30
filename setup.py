from setuptools import setup, find_packages

setup(
    name="diterm-detox",
    version="0.1.0",
    description="Terminal detox + AI bullshit detector. Flags rogue agents before they nuke your DB.",
    packages=find_packages(),
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "diterm-detox=diterm.cli:main",
        ],
    },
)
