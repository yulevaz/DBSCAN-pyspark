from pyspark.ml.param.shared import *

#Number of the rank reduction for the eigenvectors
class HasDistance(Params):

	distance = Param(Params._dummy(), "distance", "distance")

	def __init__(self):
		super(HasDistance, self).__init__()

	def setDistance(self, value):
		return self._set(distance=value)

	def getDistance(self):
		return self.getOrDefault(self.distance)

