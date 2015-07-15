import copy
import numpy as np
from sklearn.decomposition import PCA
import netCDF4
import json

from functions import extract_n_by_n

class PCA_sklearn:

	def fit(self, predictors, locations, **kwargs):

		self.locations = locations
		self.pcas = []
		self.n = predictors['n']

		for location in locations:
			raw = extract_n_by_n(predictors, location, **kwargs)
			
			#pca = PCA(n_components='mle', whiten=True)
			#pca = PCA(n_components=0.95, whiten=True)
			pca = PCA(n_components=2)
			
			pca = pca.fit(raw)
			components = pca.components_
			pca.components_ = components
			
			self.pcas.append(pca.fit(raw))

			print "pca: ", location, pca.n_components_, pca.explained_variance_ratio_


	def save(self, filename='pca.nc'):
		"""
		Write sklearn PCA parameters to a netcdf file
		"""
		max_n_components = max([pca.n_components_ for pca in self.pcas])
		n_features = self.pcas[0].components_.shape[1]

		outfile = netCDF4.Dataset(filename, 'w')

		outfile.createDimension("location", len(self.locations))
		outfile.createDimension("feature", n_features)
		outfile.createDimension("component", max_n_components)

		outfile.createVariable("location", str, ("location"))
		outfile.createVariable("n_components", "i4", ("location"))
		outfile.createVariable("components", "f4", ("location", "component", "feature"))
		outfile.createVariable("means", "f4", ("location", "feature"))
		outfile.createVariable("explained_variance_ratio", "f4", ("location", "component"))
		outfile.createVariable("noise_variance", "f4", ("location"))
		outfile.createVariable("whiten", "c", ("location"))

		id = 0
		for pca in self.pcas:
			#print id, self.locations[id], pca.n_components_, pca.components_.shape, pca.explained_variance_ratio_.shape
			outfile.variables['location'][id] = json.dumps(self.locations[id])
			outfile.variables["n_components"][id] = pca.n_components_
			outfile.variables["components"][id,:pca.n_components_] = pca.components_
			outfile.variables["means"][id] = pca.mean_
			outfile.variables["explained_variance_ratio"][id,:pca.n_components_] = pca.explained_variance_ratio_
			outfile.variables["noise_variance"][id] = pca.noise_variance_

			id += 1

		outfile.close()
		return True

	def load(self, filename='pca.nc'):
		"""
		Read sklearn PCA parameters from a netcdf file
		"""

		infile = netCDF4.Dataset(filename, 'r')

		self.locations = [json.loads(string) for string in list(infile.variables['location'])]
		self.pcas = []

		id = 0
		for location in self.locations:
			n_components = infile.variables['n_components'][id]
			components = infile.variables['components'][id]
			mean = infile.variables['means'][id]
			explained_variance_ratio = infile.variables['explained_variance_ratio'][id]
			noise_variance = infile.variables['noise_variance'][id]

			pca = PCA(n_components=n_components)
			pca.components_ = components
			pca.mean_ = mean
			pca.explained_variance_ratio_ = explained_variance_ratio
			pca.noise_variance_ = noise_variance

			self.pcas.append(pca)

			id += 1

	def transform(self, predictors, **kwargs):

		result = []

		id = 0
		for location in self.locations:
			raw = extract_n_by_n(predictors, location, **kwargs)
			result.append(self.pcas[id].transform(raw))
			id += 1

		return result