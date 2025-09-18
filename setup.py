from setuptools import setup, find_packages

setup(
    name="jbfuncs",
    version="0.0.0",
    author="Yaoshu Xie",
    author_email="jasonxie@sz.tsinghua.edu.cn",
    description="Importable functions for jobflow",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HouGroup/JobflowFunctions/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "jobflow",
    ]
)
