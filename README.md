# SimianPy

**Author**: Janahan Selvanayagam  
**E-mail**: <seljanahan@hotmail.com>  

```python
>>> import simianpy as simi
```

# Installation  
To install, clone this repo and run setuptools from the command line:
```
python setup.py install
```

To install a specific version, use pip and the appropriate release tag:
```
pip install git+file:///PATH_TO_REPO/.git@0.1.1-a1
```
OR
```
pip install git+https://github.com/jselvan/simianpy.git@0.1.1-a1
```

## For developers
to install in development mode, from the repo directory, use:
```
python setup.py install develop
```
OR 
```
pip install -e .
```

# Instructions
For help on using the code, rely on docstrings built into the code.  

For example:  
```python
>>> import simianpy as simi
>>> help(simi) 
```

Sphinx autodocumentation build in progress! See docs subdirectory

# License
The MIT License

Copyright (c) 2019, Janahan Selvanayagam  

[Read here](LICENSE.txt)
