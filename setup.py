from setuptools import setup, find_packages

setup(
    name="simianpy",
    packages=find_packages(),
    use_scm_version= {
        'write_to': 'simianpy/_version.py'
    },
    setup_requires=['setuptools_scm'],
    description="Data analysis tools designed for working with primate neuroscientific data",
    author='Janahan Selvanayagam',
    author_email='seljanahan@hotmail.com',
    keywords=['electrophysiology', 'behaviour', 'signal processing'],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'pyyaml',
        'h5py',
        'tables'
    ],
    extras_require = {
        'FULL': ['colorama','holoviews','bokeh','tqdm','ipython']
    },
    zip_safe=False
)
