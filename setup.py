from setuptools import setup, find_namespace_packages

setup(
    name="localperf",
    url="https://github.com/tboulet/Local-Python-Performance", 
    author="Timothé Boulet",
    author_email="timothe.boulet0@gmail.com",
    
    packages=find_namespace_packages(),

    version="1.0",
    license="MIT",
    description="Measure of python performance in local.",
    long_description=open('README.md').read(),      
    long_description_content_type="text/markdown",
)