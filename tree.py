import numpy as np
from matplotlib.patches import Rectangle

class BinaryNode(object):

	def __init__(self, dimension, ranges, split, left, right):

		self.dim = dimension
		self.ranges = ranges
		self.split = split
		self.left = left
		self.right = right
		self.count = 0

	def __repr__(self, level=0):

		result = '\t'*level + "[{}] {} ({}, {})\n".format(self.dim, self.split, self.left.count, self.right.count)

		if type(self.left) == self.__class__:
			result += self.left.__repr__(level+1)
		elif type(self.left) == list:
			result += '\t'*(level+1) + 'leaf[{}] '.format(len(self.left)) + repr(self.left) + '\n'

		if type(self.right) == self.__class__:
			result += self.right.__repr__(level+1)
		elif type(self.right) == list:
			result += '\t'*(level+1) + 'leaf[{}] '.format(len(self.right)) + repr(self.right) + '\n'

		return result

	def counts(self):

		if type(self.left) == self.__class__:
			left_count = self.left.counts()
		else:
			left_count = len(self.left)

		if type(self.right) == self.__class__:
			right_count = self.right.counts()
		else:
			right_count = len(self.right)

		self.left_count = left_count
		self.right_count = right_count

		return left_count + right_count

	def max_leaves(self):

		if type(self.left) == self.__class__:
			left_max = self.left.max_leaves()
		else:
			left_max = len(self.left)

		if type(self.right) == self.__class__:
			right_max = self.right.max_leaves()
		else:
			right_max = len(self.right)

		return max(left_max, right_max)


	def count_leaves(self):

		if type(self.left) == self.__class__:
			left_count = self.left.count_leaves()
		else:
			if len(self.left) > 0:
				left_count = 1
			else:
				left_count = 0

		if type(self.right) == self.__class__:
			right_count = self.right.count_leaves()
		else:
			if len(self.right) > 0:
				right_count = 1
			else:
				right_count = 0

		return left_count + right_count

	def plot(self, ax):

		if self.dim == 0:
			ax.axvline(self.split, alpha=0.1)
		else:
			ax.axhline(self.split, alpha=0.1)

		if type(self.left) == self.__class__:
			self.left.plot(ax)

		if type(self.right) == self.__class__:
			self.right.plot(ax)

	def plot_density(self, ax, color_map, threshold=0):

		if type(self.left) == self.__class__:
			self.left.plot_density(ax, color_map)
		elif len(self.left) > threshold:
			ranges = list(self.ranges)
			ranges[self.dim] = (self.ranges[self.dim][0], self.split)
			x, y = ranges[0][0], ranges[1][0]
			width = ranges[0][1]-ranges[0][0]
			height = ranges[1][1]-ranges[1][0]
			ax.add_patch(Rectangle((x, y), width, height, color=color_map.to_rgba(len(self.left)), alpha=0.7))
			#ax.text(x, y, repr(len(self.left)))

		if type(self.right) == self.__class__:
			self.right.plot_density(ax, color_map)
		elif len(self.right) > threshold:
			ranges = list(self.ranges)
			ranges[self.dim] = (self.split, self.ranges[self.dim][1])
			x, y = ranges[0][0], ranges[1][0]
			width = ranges[0][1]-ranges[0][0]
			height = ranges[1][1]-ranges[1][0]
			ax.add_patch(Rectangle((x, y), width, height, color=color_map.to_rgba(len(self.right)), alpha=0.7))

	def volume(self, depth=1):

		if type(self.left) == self.__class__:
			left_volume = self.left.volume(depth=depth+1)
		elif len(self.left) > 0:
			left_volume = 1.0/np.power(2,depth)
			#print depth, left_volume
		else:
			left_volume = 0.0

		if type(self.right) == self.__class__:
			right_volume = self.right.volume(depth=depth+1)
		elif len(self.right) > 0:
			right_volume = 1.0/np.power(2,depth)
			#print depth, right_volume
		else:
			right_volume = 0.0

		return left_volume + right_volume

	def volume_diff(self, other, depth=1):

		# Left volume difference
		# Both trees
		if type(self.left) == self.__class__ and type(other.left) == self.__class__:
			left_diff = self.left.volume_diff(other.left, depth+1)

		# Self is tree other is list of leaves
		elif type(self.left) == self.__class__ and type(other.left) == list:
			left_diff = abs(self.left.volume(depth+1) - int(len(other.left)>0) * 1.0/np.power(2,depth))

		# Self is list of leaves, other is tree
		elif type(self.left) == list and type(other.left) == self.__class__:
			left_diff = abs(other.left.volume(depth+1) - int(len(self.left)>0) * 1.0/np.power(2,depth))

		else:
			left_diff = 0.0

		# Right volume difference
		# Both trees
		if type(self.right) == self.__class__ and type(other.right) == self.__class__:
			right_diff = self.right.volume_diff(other.right, depth+1)

		# Self is tree other is list of leaves
		elif type(self.right) == self.__class__ and type(other.right) == list:
			right_diff = abs(self.right.volume(depth+1) - int(len(other.right)>0) * 1.0/np.power(2,depth))

		# Self is list of leaves, other is tree
		elif type(self.right) == list and type(other.right) == self.__class__:
			right_diff = abs(other.right.volume(depth+1) - int(len(self.right)>0) * 1.0/np.power(2,depth))

		else:
			right_diff = 0.0

		return left_diff + right_diff


class BinaryTree:

	def __init__(self, samples, ranges=None, maxdepth=100, threshold=1):

		if not ranges:
			self.ranges = zip(np.min(samples, axis=0), np.max(samples, axis=0))
		else:
			self.ranges = ranges

		self.samples = list(samples)
		self.maxdepth = maxdepth
		self.threshold = threshold
		indices = range(0, len(samples))

		self.root = self.make_tree(0, self.ranges, indices)


	def make_tree(self, dim, ranges, indices, depth=0):

		#print depth
		if depth >= self.maxdepth or len(indices) <= self.threshold:
			#print depth, len(indices), " stopping"
			return indices

		if dim >= len(ranges):
			dim = 0

		split = (ranges[dim][0] + ranges[dim][1])/2

		left_indices = []
		right_indices = []
		
		s = slice(dim, dim+1)
		for i in indices:
			if self.samples[i][s] < split:
				left_indices.append(i)
			else:
				right_indices.append(i)
		
		#print "left: ", left_indices
		#print "right: ", right_indices

		ranges_left = list(ranges)
		ranges_right = list(ranges)

		ranges_left[dim] = (ranges[dim][0], split)
		ranges_right[dim] = (split, ranges[dim][1])

		left = self.make_tree(dim+1, ranges_left, left_indices, depth+1)			
		if type(left) != list:
			left.count = len(left_indices)

		right = self.make_tree(dim+1, ranges_right, right_indices, depth+1)		
		if type(right) != list:
			right.count = len(right_indices)

		return BinaryNode(dim, ranges, split, left, right)

	def max_leaves(self):
		return self.root.max_leaves()

	def plot(self, ax):
		self.root.plot(ax)

	def plot_density(self, ax, color_map, threshold=0):
		self.root.plot_density(ax, color_map, threshold)

	def volume(self):
		return self.root.volume()

	def volume_diff(self, other):
		return self.root.volume_diff(other.root)