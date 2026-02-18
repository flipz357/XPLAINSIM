from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='xplainsim',
    version='0.9',    
    description='A Python package for explaining text similarity',
    url='https://github.com/flipz357/ExplainSimilarity',
    author='Juri Opitz, Andrianos Michail, Lucas Moeller',
    author_email='opitz.sci@gmail.com',
    license='GPLv3',
    packages=['xplain', 'xplain.spaceshaping', 'xplain.attribution', 'xplain.symbolic'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "xplain-install-amr=xplain.symbolic.parser_install:install_default_amr_model",
        ],
    },
    install_requires=["torch", 
                      "transformers[torch]", 
                      "sentence-transformers", 
                      "datasets",
                      "unidecode"], 
    classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"])
