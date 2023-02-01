import setuptools
from pkg_resources import parse_requirements

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requires = ['numpy',
            'scipy',
            'mne',
            'matplotlib',
            'pyxdf',
            'pandas',
            'pyxdf',
            'imbalanced-learn',
            'pyautogui',
            'seaborn',
            'imageio',
            'tqdm',
            'autoreject',
            'scikit-learn'
            ]

setuptools.setup(
    name="RenaAnalysis",
    version="0.0.1.dev1",
    author="ApocalyVec",
    author_email="s-vector.lee@hotmail.com",
    description="Reality Navigation Analysis Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ApocalyVec/ReNaAnalysis",
    project_urls={
        "Bug Tracker": "https://github.com/ApocalyVec/ReNaAnalysis/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=requires,

    package_data={'': ['renaanalysis/params/*.json']},
    include_package_data=True
)
