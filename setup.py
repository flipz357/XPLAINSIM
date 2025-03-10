from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='xplain',
    version='0.0.1',    
    description='A Python package for explaining text similarity',
    url='https://github.com/flipz357/ExplainSimilarity',
    author='Juri Opitz',
    author_email='opitz.sci@gmail.com',
    license='GPLv3',
    packages=['xplain'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.10",
    install_requires=["torch", "transformers", "sentence-transformers", "numpy>=1.20.1", "scipy>=1.10.1"], 
    classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"])
