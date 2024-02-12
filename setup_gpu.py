from typing import List
from setuptools import find_packages, setup


requirements: List[str] = []
with open("requirements_gpu.txt", mode="rt", encoding="utf-8") as f:
    requirements = f.readlines()

setup(
    name='hict_patterns',
    version='0.1',
    packages=list(set(['hict.patterns']).union(find_packages())),
    url='https://genome.ifmo.ru',
    license='',
    author='Vitalii Dravgelis',
    author_email='',
    description='This utilite find coordinates of structural variations breakpoints in Hi-C data',
    setup_requires=[
        'setuptools>=65.5.0',
        'wheel>=0.42.0',
    ],
    entry_points={
        'console_scripts': ['hict_patterns=hict.patterns.hict_patterns:main', 'hict=hict.entrypoint:main'],
    },
    install_requires=list(set([]).union(requirements)),
)