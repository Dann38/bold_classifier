import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bold_classifier",
    version="0.0.1",
    author="Dunn Kopylov",
    author_email="38dunn@gmail.com",
    description="getting a measure of the width of a character in a string"
                "a marked-up image of the document is fed to the input and a marked-up image is output",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "../bold_classifier"},
    packages=setuptools.find_packages(where="../bold_classifier"),
    python_requires=">=3.10"
)
