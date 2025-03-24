import brute_force_search as bfs
import eigenFaces as eigF

def bfAuth(image,dataset,radius):
	voisins_proche, distances = bfs.radius_search(dataset, image, radius)

	print("Nbr voisins: ", len(voisins_proche), "\tindex voisins dataset:", voisins_proche, "\tradius :", radius)
	print("Distances:", distances)
	if len(voisins_proche) > 0 :
		return True
	else:
		return False

def eigenAuth(image,dataset,radius):
	poidsList, poidImg = eigF.eigenFaces(image, dataset)
	voisins_proche, distances = bfs.radius_search(poidsList,poidImg,radius)

	print("Nbr voisins: ", len(voisins_proche), "\tvoisins :", voisins_proche, "\tradius :", radius)
	print("Distances:", distances)

	if len(voisins_proche) > 0:
		return True
	else:
		return False