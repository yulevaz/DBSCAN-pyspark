import numpy as np
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark.mllib.linalg.distributed import IndexedRow
from pyspark.ml.param.shared import *
from HasDistance import HasDistance
from HasRadius import HasRadius

#Previous data considered for distance calculation between them and new data
class HasConnectionsCol(HasOutputCol):

    connections = Param(Params._dummy(), "connections", "connections")

    def __init__(self):
        super(HasConnectionsCol, self).__init__()

    def set(self, value):
        return self._set(connections=value)

    def getConnectionsCol(self):
        return self.getOrDefault(self.connections)


class HasConnectivity(HasDistance, HasRadius, HasConnectionsCol):

	def __init__(self):
		super(HasConnectivity, self).__init__()

	#Convert connection CoordinateMatrix of (i-index,j-index,distance) format
	#in an array matrix
	# @param	D		CoordinateMatrix in (i-index,j-index,distance) format
	# @param	dim		Dimension of connectivity matrix
	# @return	numpy.array	Distance matrix
	def toArray(self,D,dim):
		Darr = D.entries.collect()
		Arr = np.zeros([dim,dim]) 
		np.fill_diagonal(Arr,1)
		for d in Darr:
			Arr[d.i,d.j] = d.value

		return Arr

	#Calculate connetivity matrix with IndexedRows and return a numpy.array matrix
	# @param	rddv1		First RDD with dataset
	# @param	rddv2		Second RDD with dataset
	# @param	radius		If distance(rddv1[i],rddv2[j]) < radius, then they are connected
	# @param	sc		SparkContext
	# @return	numpy.array	Connectivity matrix
	def getConnectivity(self,rddv1,rddv2,sc):
		radius = self.getRadius()
		dist = self.getDistance()
		dlist1 = rddv1.collect()
		dlist2 = rddv2.collect()
		irows1 = [IndexedRow(i,dlist1[i]) for i in range(0,len(dlist1))]
		irows2 = [IndexedRow(i,dlist2[i]) for i in range(0,len(dlist2))]
		imatrix1 = IndexedRowMatrix(sc.parallelize(irows1))
		imatrix2 = IndexedRowMatrix(sc.parallelize(irows2))
		cart = imatrix1.rows.cartesian(imatrix2.rows)
		A = cart.map(lambda x : (x[0].index, x[1].index, 1 if dist(x[0].vector,x[1].vector) <= radius else 0)
)
		return CoordinateMatrix(A)
