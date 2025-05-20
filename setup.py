from setuptools import setup, find_packages

setup(
    name='qubit_zz_coupling',
    version='0.1.0',
    author='Jared Yamaoka',
    author_email='jared.yamaoka@outlook.com',
    description='A package for simulating qubit dynamics with ZZ coupling and single TLS interactions.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'qutip',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)