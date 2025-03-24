class image:
	"""
	Simple classe permettant d'effectuer la vérité-terrain
	"""
	def __init__(self, name, version, matrix, saved):
		self.name = name
		self.version = version
		self.matrix = matrix
		self.savedGal = saved       # si la matrixe de l'image est stockée dans la gallery