from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='CTRNN',
    version='0.1.0',
    description='A package that implements Continuous Time Recurrent Neural Networks',
    long_description=readme,
    author='Madhavun Candadai',
    author_email='madvncv@gmail.com',
    url='https://github.com/madvn',
    license=license,
    packages=find_packages(),
    install_requires=['numpy'],
    scripts=['ctrnn.py']
)
