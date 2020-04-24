from setuptools import setup

setup(
    name = 'cat-class',
    version = '0.1.0',
    description = ('An image classifier employing a deep neural network '
                        'to identify pictures of cats.'),
    url = 'https://github.com/petejh/cat-class',
    author = 'Peter J. Hinckley',
    author_email = 'petejh.code@q.com',
    license = 'MIT',
    packages = ['catclass'],
    python_requires = '',
    install_requires = ['deepen', 'h5py', 'numpy'],
    zip_safe = False
)
