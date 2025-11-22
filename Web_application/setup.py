from setuptools import setup 
from Cython.Build import cythonize 
 
setup( 
    ext_modules=cythonize("services/encryption_utils.py", compiler_directives={'language_level': "3"}), 
    zip_safe=False, 
) 
