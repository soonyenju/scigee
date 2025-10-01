#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: Songyan Zhu
# Mail: soonyenju@foxmail.com
# Created Time:  2018-10-23 13:28:34
#############################################


from setuptools import setup, find_packages

setup(
	name = "scigee",
	version = "0.0.36",
	keywords = ("easy geo warpper", "atmospheric data","satellite data", "flux"),
	description = "For faster proccessing geofile",
	long_description = "Read/write and process rs/gis related data, especially atmospheric rs data.",
	license = "MIT Licence",

	url="https://github.com/soonyenju/scigee",
	author = "Songyan Zhu",
	author_email = "Songyan.Zhu@outlool.com",

	packages = find_packages(),
	include_package_data = True,
	platforms = "any",
	install_requires=[
            # "geopandas==0.4.1",
            # "scipy==1.2.1",
            # "GDAL==2.3.3",
            # "pyproj==1.9.6",
            # "Shapely==1.6.4.post1",
            # "rasterio==1.0.21",
            # "numpy==1.16.0",
			# "geopandas",
			# "rasterio"
			# "pygal",
			"geetools"

	]
)