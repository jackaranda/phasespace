import copy
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import netCDF4
import json

#from functions import extract_n_by_n

def fit(predictors, predictands, log=False, **kwargs):
	
	model = PLSRegression(n_components=2)
	try:
		model.fit(predictors, predictands)
	except:
		return None

	return model

def transform(model, predictors):

	try:
		result = model.transform(predictors)
	except:
		return None

	return result

class PLS_sklearn:

	def fit(self, predictors, predictands, locations, log=False, **kwargs):

		self.locations = locations
		self.models = []
		self.n = predictors['n']

		id = 0
		for location in locations:
			X = extract_n_by_n(predictors, location, **kwargs)
			Y = predictands[:,id]

			if log:
				Y = np.log(Y)

			#pca = PCA(n_components='mle', whiten=True)
			model = PLSRegression(n_components=2)
			
			model = model.fit(X,Y)
			#components = pca.components_
			#pca.components_ = components
			
			self.models.append(model)
			print "pls: ", location, model.score(X, Y), model.x_loadings_.shape, np.argmax(model.x_loadings_, axis=0)

			id += 1


	def transform(self, predictors, **kwargs):

		result = []

		id = 0
		for location in self.locations:
			X = extract_n_by_n(predictors, location, **kwargs)
			result.append(self.models[id].transform(X))
			id += 1

		return result


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

