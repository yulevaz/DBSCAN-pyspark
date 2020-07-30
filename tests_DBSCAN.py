from pyspark.sql import SparkSession
from HasConnectivity import HasConnectivity
from HasDenseConnectivity import HasDenseConnectivity
from HasDistance import HasDistance
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
	rdd = sc.parallelize(pts) 
	cmatrix = conn.getConnectivity(rdd,rdd,sc)
	B = conn.toArray(cmatrix,len(pts))
		
	assert 0 == np.sum(np.array(A) - B)
	
'''	
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
	rdd = sc.parallelize(pts) 
	cmatrix = conn.getConnectivity(rdd,rdd,sc)
	B = conn.toArray(cmatrix,len(pts))
		
	assert 0 == np.sum(np.array(A) - B)
'''
