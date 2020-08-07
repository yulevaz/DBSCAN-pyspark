from HasConnectivity import HasConnectivity
from HasDensity import HasDensity
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from graphframes import GraphFrame

class HasDenseConnectivity(HasConnectivity, HasDensity):

	def __init__(self):
		super(HasDenseConnectivity, self).__init__()
	
	#Generate dense connectivity graph (GraphFrame)
	# @param	cgraph		connectivity graph
	# @return	cgraph	 	dense connectivity graph
	def __densify(self,cgraph):
		nrow = cgraph.vertices.count()
		dens = self.getDensity()
		idx = cgraph.edges.rdd.map(lambda x : (x[0],x[2]))\
			.reduceByKey(lambda x,y : x + y)\
			.filter(lambda x : x[1] > dens)

		cart = cgraph.edges.rdd.cartesian(idx).filter(lambda x : (x[0][0] == x[1][0] or x[0][1] == x[1][0]))\
			.map(lambda x : x[0])
		return cgraph.vertices,cart
			
	#Calculate connetivity matrix with IndexedRows and return a numpy.array matrix
	# @param	rddv		RDD with dataset
	# @param	spark		SparkSession
	# @return	numpy.array	Connectivity matrix
	def getConnectivity(self,rddv,spark):
		conn_graph = super(HasDenseConnectivity, self).getConnectivity(rddv,spark)
		V,E = self.__densify(conn_graph)		
		return GraphFrame(V,spark.createDataFrame(E))

