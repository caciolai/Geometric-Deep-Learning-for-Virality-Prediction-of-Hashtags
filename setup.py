from setuptools import find_packages
from setuptools import setup

setup(
    name='virality',
    version='1.0.0',
    description='Geometric Deep Learning for Virality Prediction of Hashtags.',
    author='caciolai, crisostomi',
    license='MIT License',
    url='https://github.com/caciolai/Geometric-Deep-Learning-for-Virality-Prediction-of-Hashtags',
    packages=find_packages("src"),  # include all packages under src
    package_dir={"": "src"},   # tell distutils packages are under src
    python_requires='>=3.7.9',
    install_requires=[
        'dill',
        'matplotlib',
        'networkx',
        'numpy',
        'pytorch-lightning',
        'pandas',
        'scikit-learn',
        'seaborn',
        'torch',
        'torch-scatter',
        'torch-sparse',
        'torch-spline-conv',
        'torch-geometric',
        'tqdm',
        'tweepy'
    ]
)