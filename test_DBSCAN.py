from pyspark.sql import SparkSession
from HasConnectivity import HasConnectivity
from HasDenseConnectivity import HasDenseConnectivity
from HasDistance import HasDistance
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from DBSCAN import DBSCAN
import numpy as np
from pyspark import keyword_only 
import pytest
from pytest import approx

class DistMock(HasDistance):

	@keyword_only
	def __init__(self, distance=None):
		super(DistMock, self).__init__()
		kwargs = self._input_kwargs
		self.setParams(**kwargs)

	@keyword_only
	def setParams(self, distance=None):
		kwargs = self._input_kwargs
		return self._set(**kwargs) 

class ConnMock(HasConnectivity):

	@keyword_only
	def __init__(self, distance=None, radius=None):
		super(ConnMock, self).__init__()
		kwargs = self._input_kwargs
		self.setParams(**kwargs)

	@keyword_only
	def setParams(self, distance=None, radius=None):
		kwargs = self._input_kwargs
		return self._set(**kwargs) 

class DenseConnMock(HasDenseConnectivity):

	@keyword_only
	def __init__(self, distance=None, radius=None, density=None):
		super(DenseConnMock, self).__init__()
		kwargs = self._input_kwargs
		self.setParams(**kwargs)

	@keyword_only
	def setParams(self, distance=None, radius=None, density=None):
		kwargs = self._input_kwargs
		return self._set(**kwargs) 

def test_distance():

	v1 = [1,0]
	v2 = [0,1]
	d = np.sqrt(2)
	mock = DistMock(distance = lambda x,y : np.sqrt(np.sum(np.power(np.array(x)-np.array(y),2)))) 
	dr = mock.getDistance()(v1,v2)

	assert d == approx(dr)

def test_conn_as_dist():
	v1 = [1,0]
	v2 = [0,1]
	d = np.sqrt(2)
	mock = ConnMock(distance = lambda x,y : np.sqrt(np.sum(np.power(np.array(x)-np.array(y),2))),radius=1) 
	dr = mock.getDistance()(v1,v2)
	
	assert d == approx(dr)

def test_connectivity():

	spark = SparkSession.builder.getOrCreate()
	sc = spark.sparkContext

	A =	[[1,1,0,0,0,0,0,0,0]
		,[1,1,0,0,0,0,0,0,0]
		,[0,0,1,1,0,0,0,0,0]
		,[0,0,1,1,1,1,0,0,0]
		,[0,0,0,1,1,0,0,0,0]
		,[0,0,0,1,0,1,0,0,0]
		,[0,0,0,0,0,0,1,1,1]
		,[0,0,0,0,0,0,1,1,1]
		,[0,0,0,0,0,0,1,1,1]]

	pts = [[0,0],[1,0],[9,10],[10,10],[10,9],[10,11],[-5,-5],[-5,-6],[-5.5,-5.5]]	

	conn = ConnMock(distance = lambda x,y : np.sqrt(np.sum(np.abs(np.array(x)-np.array(y)))),radius=1)
	rdd = sc.parallelize(np.float_(pts).tolist())
	df = spark.createDataFrame(rdd,["_1","_2"])
	vector = VectorAssembler(inputCols=["_1","_2"],outputCol="features")
	feats = vector.transform(df).select("features")
	cmatrix = conn.getConnectivity(feats.rdd,spark)
	B = conn.toArray(cmatrix)
		
	assert 0 == np.sum(np.array(A) - B)

def test_dense_connectivity():

	spark = SparkSession.builder.getOrCreate()
	sc = spark.sparkContext

	A =	[[1,0,0,0,0,0,0,0,0,0,0]
		,[0,1,0,0,0,0,0,0,0,0,0]
		,[0,0,1,1,0,0,0,0,0,0,0]
		,[0,0,1,1,1,1,0,0,0,0,0]
		,[0,0,0,1,1,0,0,0,0,0,0]
		,[0,0,0,1,0,1,0,0,0,0,0]
		,[0,0,0,0,0,0,1,1,1,0,0]
		,[0,0,0,0,0,0,1,1,1,0,0]
		,[0,0,0,0,0,0,1,1,1,0,0]
		,[0,0,0,0,0,0,0,0,0,1,0]
		,[0,0,0,0,0,0,0,0,0,0,1]]

	pts = [[0,0],[1,0],[9,10],[10,10],[10,9],[10,11],[-5,-5],[-5,-6],[-5.5,-5.5],[-2,-2],[-5,0]]	

	conn = DenseConnMock(distance = lambda x,y : np.sqrt(np.sum(np.abs(np.array(x)-np.array(y)))),radius=1,density=2)
	rdd = sc.parallelize(np.float_(pts).tolist())
	df = spark.createDataFrame(rdd,["_1","_2"])
	vector = VectorAssembler(inputCols=["_1","_2"],outputCol="features")
	feats = vector.transform(df).select("features")
	cmatrix = conn.getConnectivity(feats.rdd,spark)
	B = conn.toArray(cmatrix)
		
	assert 0 == np.sum(np.array(A) - B)

def test_dbscan():

	spark = SparkSession.builder.getOrCreate()
	sc = spark.sparkContext

	#Note that if there is a nullvector (e.g., [0,0]) VectorAssembler will transform it in a SparseVector and possibly will break the code
	#Need to find a workaround
	pts = [[1e-10,1e-10],[1,0],[9,10],[10,10],[10,9],[10,11],[-5,-5],[-5,-6],[-5.5,-5.5],[-2,-2],[-5,0]]	
	rdd = sc.parallelize(np.float_(pts).tolist())
	df = spark.createDataFrame(rdd,["_1","_2"])
	vector = VectorAssembler(inputCols=["_1","_2"],outputCol="features")
	dbscan = DBSCAN(featuresCol="features",predictionCol="clusters",distance = lambda x,y : np.sqrt(np.sum(np.abs(np.array(x)-np.array(y)))),radius=1,density=2) 
	pipe = Pipeline(stages=[vector,dbscan])
	model = pipe.fit(df)
	res = model.transform(df)
	
	clusters = res.select("component").collect()
	
	# Organize points ids by the corresponding connected components
	dat = res.select("id","component").groupBy("component").agg(F.collect_set("id")\
			.alias("vertices")).select("vertices")
	
	# Transform them into unordered sets 
	S = dat.sort("vertices").rdd.map(lambda x : set(x["vertices"])).collect()
	# Reference to test
	ref = [{0}, {1}, {2, 3, 4, 5}, {8, 6, 7}]

	equals = True
	for r in ref:
		equals *= equals * (r in S)

	assert True == equals
