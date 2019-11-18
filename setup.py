import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
about = {}

with open(os.path.join(here, "gym_minecraft_pygame", '__version__.py'), 'r') as f:
    exec(f.read(), about)

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name=about['__title__'],
    description=about['__description__'],
    version=about['__version__'],
    author=about['__author__'],
    url=about['__url__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    keywords='environment, agent, rl, openaigym, openai-gym, gym, sapientino',
    packages=find_packages(include="gym_minecraft_pygame*"),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=["gym", "pygame", "numpy"],
    tests_require=["tox"],
    python_requires='>=3.7',
    license=about['__license__'],
    zip_safe=False
)
