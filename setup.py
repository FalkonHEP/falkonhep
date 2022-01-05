from setuptools import setup, find_packages

def get_requirements():
    with open('./requirements.txt', 'r') as f:
        requirements = [line[:-1] for line in f.readlines()]
    return requirements


setup(
    name="falkonhep",
    version="1.0.0",
    description="Falkon for High Energy Physics",
    python_requires='~=3.8',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages(),
    install_requires=get_requirements(),
    include_package_data=True,
)