import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="x1rl", # Replace with your own username
    version="0.0.1",
    author="Gyuyoul Kim",
    author_email="gyuyoul@gmail.com",
    description="rl frameworks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GyuyoulKim/RLSandbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)