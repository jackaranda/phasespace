import sys
import logging
import copy
import numpy as np
import datetime
from dateutil import parser
from shapely.geometry import mapping, shape


import netCDF4

sys.path.append('../pycdm2/')
import pycdm

import pls_sklearn


def open_uris(config):

	result = []

	for source in config:
		uri = source['uri']

		try:
			ds = pycdm.open(uri)
		except:
			logging.error('failed to open uri {}'.format(uri))
			continue

		this = {}
		this['uri'] = uri
		this['ds'] = ds
		this['variables'] = []

		for variable, levels in source['variables'].items():

			try:
				field = pycdm.Field(ds.root.variables[variable])
			except:
				logging.warning("variable {} not found".format(variable))
				continue

			this['variables'].append({'name':variable, 'field': pycdm.Field(ds.root.variables[variable]), 'levels':levels})

		result.append(this)

	return result

def features_to_centroids(features):

	result = []
	for feature in features['features']:

		s = shape(feature['geometry'])
		result.append(s.centroid)

	return result


def load_pipeline(config):

	result = []

	for component in config['pipeline']:
		this = copy.copy(component)

		try:
			function = eval(component['function'])
		except:
			logging.error("Cannot map function {} to real function".format(component["function"]))
			continue

		this['function'] = function
		result.append(this)
		
	return result

def n_by_n(predictors, location, n=0, startdate=None, enddate=None, months=None):

	# Get latitude and longitude
	longitude, latitude = location.x, location.y
	print latitude, longitude

	nbyn = pow(n*2+1,2)

	# Convert dates to datetimes
	try:
		startdate = parser.parse(startdate)
		enddate = parser.parse(enddate)
	except:
		pass

	# Do initial count of columns and rows
	count = 0

	for source in predictors:
		for variable in source['variables']:
			
			levels = variable['levels']

			if len(levels) == 0:
				count += nbyn
			else:
				count += len(levels) * nbyn

			field = variable['field']

			if startdate and enddate:
				field.subset(time=(startdate, enddate))

			realtimes = field.realtimes

			if type(months) == list:
				time_select = np.nonzero([time.month in months for time in field.realtimes])[0]
				time_steps = time_select.shape[0]
			else:
				time_steps = len(realtimes)

	print "will extract {} columns".format(count)
	print "will extract {} rows".format(time_steps)

	result = np.empty((time_steps, count))

	column = 0
	for source in predictors:
		for variable in source['variables']:

			field = variable['field']
			times = field.realtimes

			for level in levels:
				slices = list(field.reversemap(latitude=latitude, longitude=longitude, level=level))
				
				mod_slices = copy.copy(slices)
				mod_slices[field.level_dim] = slices[field.level_dim]


				mod_slices[-1] = slice(slices[-1].start - n, slices[-1].stop + n)
				mod_slices[-2] = slice(slices[-2].start - n, slices[-2].stop + n)

				slices = tuple(mod_slices)
				print nbyn
				print slices
				print field[slices].shape
				print result[:,column:column + nbyn].shape
				#print field[slices].reshape(time_steps,nbyn).shape

				try:
					result[:,column:column + nbyn] = field[slices].reshape(time_steps,nbyn)
				except:
					logging.error("n_by_n failed for this location")
					result[:,column:column + nbyn] = np.nan

				column += nbyn

	return result

