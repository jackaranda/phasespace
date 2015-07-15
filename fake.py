import sys
import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


import tree

parser = argparse.ArgumentParser()
parser.add_argument('--dims', type=int, default=2)
parser.add_argument('--samples', type=int, default=1000000)
parser.add_argument('-d', '--maxdepth', type=int, default=50)
parser.add_argument('-t', '--threshold', type=int, default=100)
args = parser.parse_args()

maxdepth = args.maxdepth
threshold = args.threshold
samples_count = args.samples
dim_count = args.dims

samples = np.ndarray((samples_count, dim_count))
#samples = np.random.rand(samples_count, dim_count) - 0.5

samples[:samples_count/2] = np.random.randn(samples_count/2, dim_count) + 2
samples[samples_count/2:] = np.random.randn(samples_count/2, dim_count) - 2


samples2 = np.ndarray((samples_count, dim_count))
#samples = np.random.rand(samples_count, dim_count) - 0.5

samples2[:samples_count/2] = np.random.randn(samples_count/2, dim_count) - 2
samples2[samples_count/2:] = np.random.randn(samples_count/2, dim_count) + 2

ranges = zip(np.min(samples, axis=0), np.max(samples, axis=0))
indices = range(0, samples_count)

print "initial ranges = ", ranges

#fig = plt.fig = plt.figure(figsize=(10,10))
#plt.scatter(samples[:,0], samples[:,1],s=1)
#plt.show()

samples = list(samples)

def make_tree(dim, ranges, samples, indices, depth=0):

#	print depth
	if depth > maxdepth or len(indices) <= threshold:
		#print depth, len(indices), " stopping"
		return indices

	if dim >= len(ranges):
		dim = 0

	split = (ranges[dim][0] + ranges[dim][1])/2
	#print depth, dim, split

	left_indices = []
	right_indices = []
	
	s = slice(dim, dim+1)
	for i in indices:
		if samples[i][s] < split:
			left_indices.append(i)
		else:
			right_indices.append(i)
	
	#print "left: ", left_indices
	#print "right: ", right_indices

	ranges_left = list(ranges)
	ranges_right = list(ranges)

	ranges_left[dim] = (ranges[dim][0], split)
	ranges_right[dim] = (split, ranges[dim][1])


	left = make_tree(dim+1, ranges_left, samples, left_indices, depth+1)			
	if type(left) != list:
		left.count = len(left_indices)

	right = make_tree(dim+1, ranges_right, samples, right_indices, depth+1)		
	if type(right) != list:
		right.count = len(right_indices)

	return tree.binary_node(dim, ranges, split, left, right)


tree = make_tree(0, ranges, samples, indices)

#print tree.counts()
#print tree
#print tree.count_children()

vmin = 0
vmax = threshold
mappable = ScalarMappable(cmap='Blues')
mappable.set_array(np.arange(vmin,vmax,0.1))
mappable.set_clim((vmin,vmax))


fig = plt.fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.scatter(np.array(samples)[:,0], np.array(samples)[:,1], s=1, alpha=0.4, color='black')
tree.plot_density(ax, mappable)
plt.savefig("fake_{}samples_threshold{}_maxdepth{}.png".format(samples_count, threshold, maxdepth))
plt.show()

