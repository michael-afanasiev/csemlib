from setuptools import setup, find_packages

setup(
    name='csemlib',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['click', 'numpy'],
    entry_points='''
    [console_scripts]
    csem=csemlib.csemlib:cli
    '''
)
