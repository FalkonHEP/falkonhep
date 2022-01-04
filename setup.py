from setuptools import setup, find_packages

setup(
    name="falkonhep",
    version="1.0.0",
    description="Falkon for High Energy Physics",
    python_requires='~=3.8',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages(),
    include_package_data=True,
)