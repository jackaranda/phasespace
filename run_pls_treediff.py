import json
import sys
import numpy as np
import datetime
import functions
from pls_sklearn import PLS_sklearn
from pca_sklearn import PCA_sklearn
from tree import BinaryTree

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import seaborn


config = json.loads(open(sys.argv[1]).read())
method = 'pls'

predictors = functions.open(config['predictors'])
startdate = datetime.datetime(1980,1,1)
enddate = datetime.datetime(2010,12,31)

predictands = functions.open(config['predictands'])
path, config = predictands['sources'].items()[0]
field = config['fields']['pr']
slices = [slice(None)]*2
start_index = field.reversemap(time=startdate)[field.time_dim].start			
end_index = field.reversemap(time=enddate)[field.time_dim].start
slices[field.time_dim] = slice(start_index, end_index+1)
predictand_data = field.variables[0][slices]
print 'predictand_data.shape = ', predictand_data.shape

locations = predictands['sources'][predictands['sources'].keys()[0]]['fields']['pr'].features()

locations = locations['features'][20:22]

if method == 'pls':
	pls = PLS_sklearn()
	pls.fit(predictors, predictand_data, locations, log=True, startdate=startdate, enddate=enddate)
	reduced_predictors = pls.transform(predictors, startdate=startdate, enddate=enddate)
	reduced_predictors_subset1 = pls.transform(predictors, startdate=startdate, enddate=enddate, months=[6,7,8])
	reduced_predictors_subset2 = pls.transform(predictors, startdate=startdate, enddate=enddate, months=[12,1,2])

if method == 'pca':
	pca = PCA_sklearn()
	pca.fit(predictors, locations, startdate=startdate, enddate=enddate)
	reduced_predictors = pca.transform(predictors, startdate=startdate, enddate=enddate)
	reduced_predictors_subset1 = pca.transform(predictors, startdate=startdate, enddate=enddate, months=[6,7,8])
	reduced_predictors_subset2 = pca.transform(predictors, startdate=startdate, enddate=enddate, months=[12,1,2])



id = 20
for pred in reduced_predictors:

	ranges = zip(np.min(pred, axis=0), np.max(pred, axis=0))

	tree = BinaryTree(pred, ranges=ranges, maxdepth=10)
	tree_subset1 = BinaryTree(reduced_predictors_subset1[id-20], ranges=ranges, maxdepth=10)
	tree_subset2 = BinaryTree(reduced_predictors_subset2[id-20], ranges=ranges, maxdepth=10)
	print "this location is at ", locations[id-20]

	full_volume = tree.volume()
	subset1_volume = tree_subset1.volume()
	subset2_volume = tree_subset2.volume()
	volume_overlap = full_volume - tree_subset1.volume_diff(tree_subset2)

	fig = plt.fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111)

	#vmin = 0
	#vmax = tree.max_leaves()
	#print "max leaves = ", vmax
	#mappable = ScalarMappable(cmap='Greys')
	#mappable.set_array(np.arange(vmin,vmax,0.1))
	#mappable.set_clim((vmin,vmax))
	#tree.plot_density(ax, mappable, 10)

	#vmin = 0
	#vmax = tree_subset.max_leaves()
	#print "max leaves = ", vmax
	#mappable = ScalarMappable(cmap='Blues')
	#mappable.set_array(np.arange(vmin,vmax,0.1))
	#mappable.set_clim((vmin,vmax))
	#tree_subset.plot_density(ax, mappable,3)


	ax.scatter(np.array(tree.samples)[:,0], np.array(tree.samples)[:,1], c='grey', s=5, alpha=0.3)
	ax.scatter(np.array(tree_subset1.samples)[:,0], np.array(tree_subset1.samples)[:,1], c='blue', s=5, alpha=0.9)
	ax.scatter(np.array(tree_subset2.samples)[:,0], np.array(tree_subset2.samples)[:,1], c='red', s=5, alpha=0.9)
	ax.scatter(np.array(tree.samples)[:,0], np.array(tree.samples)[:,1], s=predictand_data[:,id]*2-1, c='green', alpha=0.8)
	#tree_subset.plot_density(ax, blue_mappable)
	plt.title("total = {}, subset1 = {}, subset2 = {}, overlap = {}".format(full_volume, subset1_volume, subset2_volume, volume_overlap))
	plt.savefig("{}_{}.notemp.png".format(method, functions.location_hashable(locations[id-20])))
	#plt.show()

	id += 1
