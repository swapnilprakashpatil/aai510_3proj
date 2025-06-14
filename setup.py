from setuptools import setup, find_packages

setup(
    name='patient_appointment_analysis',
    version='0.1.0',
    author='Swapnil Patil, Kevin Hooman, Keith Holder',
    author_email='spatil@sandiego.edu',
    description='A project for predicting and analyzing patient appointments show/no show outcomes.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'plotly',
        'nltk',
        'transformers',
        'streamlit',
        'jupyter',
        'pytest'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)