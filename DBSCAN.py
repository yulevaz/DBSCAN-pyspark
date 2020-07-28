import spark_eigen
import numpy as np
from pyspark import SparkContext
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
from pyspark.mllib.linalg.distributed import IndexedRow
from pyspark.mllib.linalg import DenseMatrix
from pyspark.ml.pipeline import Estimator, Model, Pipeline
from pyspark.ml.param.shared import *
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable 
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import FloatType
from pyspark.sql import Row
from pyspark import keyword_only  
from pyspark.sql.functions import col
import HasConnectivity
import operator
import functools

#Number of the rank reduction for the eigenvectors
class HasDensity(Params):

    density = Param(Params._dummy(), "density", "density")

    def __init__(self):
        super(HasDensity, self).__init__()

    def setDensity(self, value):
        return self._set(density=value)

    def getDensity(self):
        return self.getOrDefault(self.density)

#Previous data considered for distance calculation between them and new data
class HasPrevdata(Params):

    prevdata = Param(Params._dummy(), "prevdata", "prevdata")

    def __init__(self):
        super(HasPrevdata, self).__init__()

    def setPrevdata(self, value):
        return self._set(prevdata=value)

    def getPrevdata(self):
        return self.getOrDefault(self.prevdata)

#Estimator of spectral clustering for PySpark
class DBSCAN( Estimator, HasFeaturesCol, HasOutputCol,
			HasPredictionCol, HasRadius, HasDistance, 
			HasDensity, HasConnectivity.HasConnectivity,
			# Credits https://stackoverflow.com/a/52467470
			# by https://stackoverflow.com/users/234944/benjamin-manns
			DefaultParamsReadable, DefaultParamsWritable):

	@keyword_only
	def __init__(self, featuresCol=None, outputCol=None, radius=None, density=1):
		super(SpectralClustering, self).__init__()
		kwargs = self._input_kwargs
		self.setParams(**kwargs)

	 # Required in Spark >= 3.0
	def setFeaturesCol(self, value):
		"""
		Sets the value of :py:attr:`featuresCol`.
		""" 
		return self._set(featuresCol=value)

	# Required in Spark >= 3.0
	def setPredictionCol(self, value):
		"""
		Sets the value of :py:attr:`predictionCol`.
		"""
		return self._set(predictionCol=value)

	@keyword_only
	def setParams(self, featuresCol=None, predictionCol=None, radius=None, density=1):
		kwargs = self._input_kwargs
		return self._set(**kwargs) 

	def _getConnections(self,indexedMatrix):
			

	def _fit(self, dataset):
		sc = SparkContext.getOrCreate()
	
		x = dataset.select(self.getFeaturesCol())
		rddv = x.rdd.map(list)
		#calculate distance amtrix
		rad = self.getRadius()
		Aarr = self.getConnectivity(rddv,rddv,radius,sc)
		return DBSCANModel(featuresCol=self.getFeaturesCol(), predictionCol=self.getPredictionCol(), radius=self.getRadius(), prevdata=rddv)
		
#Transformer of spectral clustering for pySpark
class DBSCANModel(Model, HasFeaturesCol, HasPredictionCol,
			HasPrevdata, HasDistance,HasRadius,HasDensity,
			DefaultParamsReadable, DefaultParamsWritable):

	@keyword_only
	def __init__(self, featuresCol=None, predictionCol=None, radius=None, density=1, prevdata=None):
		super(SpectralClusteringModel, self).__init__()
		kwargs = self._input_kwargs
		self.setParams(**kwargs)

	@keyword_only
	def setParams(self, featuresCol=None, predictionCol=None, radius=None, density=1, prevdata=None):
		kwargs = self._input_kwargs
		return self._set(**kwargs) 

	def _transform(self, dataset):
		sc = SparkContext.getOrCreate()

		#Get spectral clustering projecction
		P = self.getProjection()
		#Get data
		x = dataset.select(self.getFeaturesCol())
		rdd2 = x.rdd.map(list)
		#Get data adopted to calculate projection
		rdd = self.getPrevdata()
		#Calculate distance between new data and "training one"
		Aarr = self._dist_matrix(rdd,rdd2,sc)
		Arm = RowMatrix(sc.parallelize(Aarr))
		#Transform new data
		result = Arm.multiply(P)
		df = result.rows.map(lambda x : Row(x.toArray().tolist())).toDF()
		return df.withColumnRenamed("_1","projection")
