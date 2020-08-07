import numpy as np
from pyspark.ml.pipeline import Model, Estimator
from pyspark.ml.param.shared import *
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable 
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark import keyword_only  
from HasDenseConnectivity import HasDenseConnectivity
from HasDistance import HasDistance
from HasRadius import HasRadius
from HasDensity import HasDensity

'''#Previous data considered for distance calculation between them and new data
class HasPrevdata(Params):

    prevdata = Param(Params._dummy(), "prevdata", "prevdata")

    def __init__(self):
        super(HasPrevdata, self).__init__()

    def setPrevdata(self, value):
        return self._set(prevdata=value)

    def getPrevdata(self):
        return self.getOrDefault(self.prevdata)
'''


#Estimator of DBSCAN for PySpark
class DBSCAN( Estimator, HasPredictionCol, HasDenseConnectivity,
			# Credits https://stackoverflow.com/a/52467470
			# by https://stackoverflow.com/users/234944/benjamin-manns
			DefaultParamsReadable, DefaultParamsWritable):

	@keyword_only
	def __init__(self, featuresCol=None, predictionCol=None, distance=None, radius=None, density=1):
		super(DBSCAN, self).__init__()
		kwargs = self._input_kwargs
		self.setParams(**kwargs)

	# Required in Spark >= 3.0
	def setPredictionCol(self, value):
		"""
		Sets the value of :py:attr:`predictionCol`.
		"""
		return self._set(predictionCol=value)

	@keyword_only
	def setParams(self, featuresCol=None, predictionCol=None, distance=None, radius=None, density=1):
		kwargs = self._input_kwargs
		return self._set(**kwargs) 


	def _fit(self, dataset):
		return DBSCANModel(featuresCol=self.getFeaturesCol(), predictionCol=self.getPredictionCol(),
					distance=self.getDistance(),radius=self.getRadius(), density=self.getDensity())

#Transformer of spectral clustering for pySpark
class DBSCANModel(Model, HasPredictionCol, HasDenseConnectivity, 
			DefaultParamsReadable, DefaultParamsWritable):

	@keyword_only
	def __init__(self, featuresCol=None, predictionCol=None, distance=None, radius=None, density=1):
		super(DBSCANModel, self).__init__()
		kwargs = self._input_kwargs
		self.setParams(**kwargs)

	@keyword_only
	def setParams(self, featuresCol=None, predictionCol=None, distance=None, radius=None, density=1):
		kwargs = self._input_kwargs
		return self._set(**kwargs) 

	def _transform(self, dataset):
		spark = SparkSession.builder.getOrCreate()	
		x = dataset.select(self.getFeaturesCol())
		graph = self.getConnectivity(x.rdd,spark)
		spark.sparkContext.setCheckpointDir("/tmp/DBSCAN-connected-components")
		df = graph.connectedComponents()
		return df
