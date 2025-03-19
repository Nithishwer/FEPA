from setuptools import setup, find_packages

setup(
    name="fepa",
    version="0.1.0",
    description="A package for analyzing the binding pocket for Free Energy simulations",
    author="Nithishwer Mouroug Anand",
    author_email="nithishwer@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license=open("LICENSE").read(),
    python_requires="==3.10.0",
    install_requires=[
        "pandas>=2.2.3",
        "pensa>=0.6.0",
        "seaborn>=0.13.2",
    ],
    packages=find_packages(),
    url="https://github.com/Nithishwer/FEPA",
    project_urls={
        "Homepage": "https://github.com/Nithishwer/FEPA",
        "Repository": "https://github.com/Nithishwer/FEPA",
    },
)
