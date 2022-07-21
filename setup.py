from setuptools import setup, find_packages

setup(
    name='d2sm',
    version='latest',
    description='Deep Semantic Statistics Matching w/ PyTorch',
    packages=find_packages(exclude=('tests', 'doc')),
    author='Kangfu Mei',
    author_email='mikumkf@gmail.com',
    install_requires=["torch", "torchvision"],
    url='https://github.com/MKFMIKU/d2sm.git',
)