from pyspark.ml.param.shared import *

#Number of the rank reduction for the eigenvectors
class HasDensity(Params):

    density = Param(Params._dummy(), "density", "density")

    def __init__(self):
        super(HasDensity, self).__init__()

    def setDensity(self, value):
        return self._set(density=value)

    def getDensity(self):
        return self.getOrDefault(self.density)

