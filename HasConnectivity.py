import numpy as np
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark.mllib.linalg.distributed import IndexedRow
from pyspark.ml.param.shared import *
from pyspark import keyword_only  
from pyspark.sql import *
from pyspark.sql import Row
from graphframes import GraphFrame
from HasDistance import HasDistance
from HasRadius import HasRadius

'''
#Previous data considered for distance calculation between them and new data
class HasConnectionsCol(HasOutputCol):

    connections = Param(Params._dummy(), "connections", "connections")

    def __init__(self):
        super(HasConnectionsCol, self).__init__()

    def set(self, value):
        return self._set(connections=value)

    def getConnectionsCol(self):
        return self.getOrDefault(self.connections)
'''

class HasConnectivity(HasFeaturesCol, HasDistance, HasRadius):

	def __init__(self):
		super(HasConnectivity, self).__init__()
		kwargs = self._input_kwargs
		self.setParams(**kwargs)
	 
	# Required in Spark >= 3.0
	def setFeaturesCol(self, value):
		"""
		Sets the value of :py:attr:`featuresCol`.
		""" 
		return self._set(featuresCol=value)

	@keyword_only
	def setParams(self, featuresCol=None, predictionCol=None, distance=None, radius=None, density=1):
		kwargs = self._input_kwargs
		return self._set(**kwargs) 


	#Convert connection GraphFrame of (i-index,j-index,connected) format
	#in an array matrix
	# @param	D		GraphFrame in (i-index,j-index,connected) format
	# @return	numpy.array	Distance matrix
	def toArray(self,D):
		Darr = D.edges.collect()
		dim = D.vertices.count()
		Arr = np.zeros([dim,dim]) 
		np.fill_diagonal(Arr,1)
		for d in Darr:
			Arr[d[0],d[1]] = d[2]

		return Arr

	#Calculate connetivity matrix with IndexedRows and return a GraphX
	# @param	rddv		RDD with dataset
	# @param	spark		SparkSession
	# @return	numpy.array	Connectivity matrix
	def getConnectivity(self,rddv,spark):
		sc = spark.sparkContext
		radius = self.getRadius()
		dist = self.getDistance()
		dlist = rddv.collect()
		featurecol = self.getFeaturesCol()
		irows = [IndexedRow(i,dlist[i][featurecol].toArray()) for i in range(0,len(dlist))]
		imatrix = IndexedRowMatrix(sc.parallelize(irows))
		cart = imatrix.rows.cartesian(imatrix.rows)

		rows = Row("id","vector")
		usr_row = [rows(i,np.float_(x).tolist()) for i,x in enumerate(dlist)]
		verts = spark.createDataFrame(usr_row)
		A = cart.filter(lambda x : dist(x[0].vector,x[1].vector) <= radius).map(lambda x : (x[0].index, x[1].index, 1))
		edges = spark.createDataFrame(A,['src','dst','connected'])
		return GraphFrame(verts,edges)
