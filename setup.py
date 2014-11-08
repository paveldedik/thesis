# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

from pip.req import parse_requirements


# read requirements
parsed_req = parse_requirements('requirements.txt')
install_req = [str(line.req) for line in parsed_req]


setup(
    name='models',
    version='1.0',
    author=u'Pavel Dedík',
    py_modules=['manage'],
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=install_req,
    zip_safe=False,
)
