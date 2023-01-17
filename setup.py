import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agentFIL",
    verison="0.1",
    author="Kiran Karra, Tom Mellan",
    author_email="kiran.karra@gmail.com, t.mellan@imperial.ac.uk",
    description="Agent based model for Filecoin Economy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://github.com/protocol/filecoin-agent-twin",
        "Source": "https://github.com/protocol/filecoin-agent-twin",
    },
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.23.1", "pandas>=1.4.3", "requests>=2.28.1"],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)