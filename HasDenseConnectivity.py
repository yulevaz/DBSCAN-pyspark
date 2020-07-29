from HasConnectivity import HasConnectivity
from HasDensity import HasDensity

class HasDenseConnectivity(HasConnectivity, HasDensity):

	def __init__(self):
		super(HasDenseConnectivity, self).__init__()
	
	#Generate dense connectivity matrix
	# @param	CMatrix		CoordinateMatrix
	# @return	CMatrix		CoordinateMatrix with dense connectivity matrix 
		
	#Calculate connetivity matrix with IndexedRows and return a numpy.array matrix
	# @param	rddv1		First RDD with dataset
	# @param	rddv2		Second RDD with dataset
	# @param	radius		If distance(rddv1[i],rddv2[j]) < radius, then they are connected
	# @param	sc		SparkContext
	# @return	numpy.array	Connectivity matrix
	def getConnectivity(self,rddv1,rddv2,sc):
		dens = self.getDensity()
		rdd = super(HasDenseConnectivity, self).getConnectivity(rddv1,rddv2,sc)
		
		return rdd

