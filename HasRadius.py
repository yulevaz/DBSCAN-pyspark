from pyspark.ml.param.shared import *

#Radius of the DBSCAN to transform new data
class HasRadius(Params):

    radius = Param(Params._dummy(), "radius", "radius")

    def __init__(self):
        super(HasRadius, self).__init__()

    def setRadius(self, value):
        return self._set(radius=value)

    def getRadius(self):
        return self.getOrDefault(self.radius)

