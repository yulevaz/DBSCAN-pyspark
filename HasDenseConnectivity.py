from HasConnectivity import HasConnectivity
from HasDensity import HasDensity
from pyspark.mllib.linalg.distributed import CoordinateMatrix

class HasDenseConnectivity(HasConnectivity, HasDensity):

	def __init__(self):
		super(HasDenseConnectivity, self).__init__()
	
	#Generate dense connectivity matrix
	# @param	cmatrix		CoordinateMatrix
	# @return	cmatrix	 	CoordinateMatrix with dense connectivity matrix 
	def __densify(self,cmatrix):
		nrow = cmatrix.numRows()
		dens = self.getDensity()
		idx = cmatrix.entries.map(lambda x : (x.i,x.value))\
			.reduceByKey(lambda x,y : x + y)\
			.map(lambda x : x[0] if x[1] > dens else None)\
			.filter(lambda x : x != None)

		cart = cmatrix.entries.cartesian(idx).map(lambda x : x[0] if (x[0].j == x[1] or x[0].i == x[1]) and x[0].value == 1 else None)\
			.filter(lambda x : x != None)
			
		return CoordinateMatrix(cart)
			
	#Calculate connetivity matrix with IndexedRows and return a numpy.array matrix
	# @param	rddv1		First RDD with dataset
	# @param	rddv2		Second RDD with dataset
	# @param	radius		If distance(rddv1[i],rddv2[j]) < radius, then they are connected
	# @param	sc		SparkContext
	# @return	numpy.array	Connectivity matrix
	def getConnectivity(self,rddv1,rddv2,sc):
		cmatrix = super(HasDenseConnectivity, self).getConnectivity(rddv1,rddv2,sc)
		R = self.__densify(cmatrix)			
		return R

