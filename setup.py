from setuptools import setup, find_packages


setup(name='plotly_utils',
      version='v0.0.2',
      description='Utilities for statistical plots with Plotly',
      url='https://github.com/dtoniolo/SDMTSA-Utils',
      author='Davide Toniolo',
      author_email='d.toniolo2@campus.unimib.it',
      licence='MIT',
      packages=find_packages(include=['plotly_utils', 'plotly_utils.*']))
