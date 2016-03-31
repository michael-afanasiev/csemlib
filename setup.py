from setuptools import setup, find_packages

setup(
    name='csemlib',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    entry_points='''
    [console_scripts]
    csem=csemlib.csemlib:cli
    ''', requires=['click']
)
