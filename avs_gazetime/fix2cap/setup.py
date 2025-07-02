import setuptools


setuptools.setup(
    name="fix2cap",
    version="0.1.0",
    author="Philip Sulewski",
    author_email="psulewski@uos.de",
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "Pillow",
        "spacy",
        "PyQt5"
    ],
)

