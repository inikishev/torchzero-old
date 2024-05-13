from setuptools import setup

setup(
   name='torchzero',
   version='0.1',
   description='0th order (derivative-free) optimizers for pytorch that fully support the optimizer API and other things.',
   author='Big Chungus',
   author_email='nkshv2@gmail.com',
   packages=['torchzero'],  #same as name
   install_requires=['torch'], #external packages as dependencies
)
