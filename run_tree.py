import json
import sys
import numpy as np
import datetime
import functions
from pca_sklearn import PCA_sklearn
from tree import BinaryTree

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import seaborn


config = json.loads(open(sys.argv[1]).read())

predictors = functions.open(config['predictors'])
startdate = datetime.datetime(1980,1,1)
enddate = datetime.datetime(2010,12,31)

predictands = functions.open(config['predictands'])
path, config = predictands['sources'].items()[0]
field = config['fields']['tasmax']
slices = [slice(None)]*2
start_index = field.reversemap(time=startdate)[field.time_dim].start			
end_index = field.reversemap(time=enddate)[field.time_dim].start
slices[field.time_dim] = slice(start_index, end_index+1)
predictand_data = field.variables[0][slices]
print 'predictand_data.shape = ', predictand_data.shape

locations = predictands['sources'][predictands['sources'].keys()[0]]['fields']['tasmax'].features()
#print locations

#for location in locations['features']:
#	print location

locations = locations['features'][20:22]

pca = PCA_sklearn()
pca.fit(predictors, locations, startdate=startdate, enddate=enddate)
#pca.save('pca.nc')
#pca.load('pca.nc')
reduced_predictors = pca.transform(predictors, startdate=startdate, enddate=enddate)

vmin = 0
vmax = 50
mappable = ScalarMappable(cmap='Blues')
mappable.set_array(np.arange(vmin,vmax,0.1))
mappable.set_clim((vmin,vmax))

id = 20
for pred in reduced_predictors:

	tree = BinaryTree(pred, maxdepth=10)

	fig = plt.fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111)
#	ax.scatter(np.array(tree.samples)[:,0], np.array(tree.samples)[:,1], s=1, alpha=0.4, color='black')
	values = np.ma.masked_less(predictand_data[:,id], 0.1)
#	ax.scatter(np.array(tree.samples)[:,0], np.array(tree.samples)[:,1], c='grey', s=5, alpha=0.3)
#	ax.scatter(np.array(tree.samples)[:,0], np.array(tree.samples)[:,1], c=values, s=values, alpha=0.7)
	tree.plot_density(ax, mappable)
	plt.show()

	id += 1

#fig = plt.fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111)
#ax.scatter(reduced_predictors[:,0], reduced_predictors[:,1], s=1, alpha=0.4, color='black')
#fig.show()
#for location in locations:
#	print location

#pcas = functions.pca_fit(predictors, locations, 1)
#for location_string, pca in pcas.items():
#	print location_string
#	print pca.n_components_
#	print pca.components_.shape
#	print pca.mean_.shape

#functions.pca_save(pcas, 'pca.nc')
#pcas = functions.pca_load('pca.nc')
#print "I've read back ", pcas.keys()


#pca_predictors = functions.pca_transform(predictors, pcas, locations)

#for location_string, data in pca_predictors.items():
#	print location_string, data.shape

#predictors[location_string] = {}
#predictors[location_string]['raw'] = functions.extract_n_by_n(predictors, locations, 0)
#predictors[location_string]['pca'] = functions.calculate_pca(predictors[location_string]['raw'])
#predictors[location_string]['']



