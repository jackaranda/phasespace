import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import seaborn

import tree

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('-d', '--maxdepth', type=int, default=50)
parser.add_argument('-t', '--threshold', type=int, default=100)
args = parser.parse_args()

maxdepth = args.maxdepth
threshold = args.threshold

infile = open(args.source)
inlines = infile.readlines()

dim_count = int(inlines[0].split()[0])
dim_count = 2
samples_count = len(inlines) - 1

print 'dim_count = ', dim_count
print 'samples_count = ', samples_count
print 'threshold = ', threshold
samples = np.ndarray((samples_count, dim_count))

index = 0
for line in inlines[1:]:
	try:
		vals = [float(word) for word in line.split()]
#		samples[index,:] = vals[:-1]
		samples[index,:] = vals[:dim_count]
	except:
		pass
	samples[index,:] += (np.random.rand(dim_count)-0.5)*0.1
	#print samples[index]
	index += 1

#fig = plt.fig = plt.figure(figsize=(10,10))
#plt.scatter(samples[:,0], samples[:,1],s=1)
#plt.show()

ranges = zip(np.min(samples, axis=0), np.max(samples, axis=0))
print "initial ranges = ", ranges
samples = list(samples)
indices = range(0, samples_count)

def make_tree(dim, ranges, samples, indices, depth=0):

	#print depth
	if depth >= maxdepth or len(indices) <= threshold:
		#print depth, len(indices), " stopping"
		return indices

	if dim >= len(ranges):
		dim = 0

	split = (ranges[dim][0] + ranges[dim][1])/2
	#prinfrom matplotlib.cm import ScalarMappablet depth, dim, split

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
leaves = tree.count_leaves()
volume = tree.volume()
print "leaves = ", leaves
print "volume = ", volume

vmin = 0
vmax = threshold
mappable = ScalarMappable(cmap='Blues')
mappable.set_array(np.arange(vmin,vmax,0.1))
mappable.set_clim((vmin,vmax))


fig = plt.fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.scatter(np.array(samples)[:,0], np.array(samples)[:,1], s=1, alpha=0.4, color='black')
tree.plot_density(ax, mappable)
plt.title("leaves = {}, volume = {}".format(leaves, volume))
plt.savefig("brucedata_{}samples_threshold{}_maxdepth{}.png".format(samples_count, threshold, maxdepth))
plt.show()



