import matplotlib.pyplot as plt
import dataLoad as load
import numpy as np

#load.dataLoad(3)

gallery = load.gallery
probes = load.probes

"""
:param :    image:      matrice de l'image
			dataset:    liste de matrices
			
:objectifs: effectuer la méthode EigenFaces par le biais d'une ACP efficace
			et l'analyse des vecteurs principaux

:returns:   poidsList :
			poidImg:    
"""
def eigenFaces(image, dataset):
	galV = np.array([l.flatten() for l in dataset])     #linéarisation des images en vecteurs 1*22500
	moyenne = np.mean(galV, axis=0)
	galVY = galV - moyenne    # centrage des données (galerie - visage moyen) phi

	print("Enclenchement du PCA efficace...", end='')
	covDT = np.cov(galVY,rowvar="True")     # Obtention d'une matrice de covariance (n*n)
	print("Covariance créee...")
	eigFacValPro, eigenFacVecPro = np.linalg.eigh(covDT)  # Calcul des vecteurs propres vi de cov(D^T)

	eigenPrincip = []
	for i in range(0,len(eigenFacVecPro)):          # Calcul des vecteurs principaux wi de D
		vecPrincip = np.dot(galVY.T, eigenFacVecPro[i])     # Obtention de vecteurs 1*22500
		norm = np.linalg.norm(vecPrincip)
		if norm != 0:
			vecPrincip = vecPrincip/norm            # Normalisation des vecteurs principaux
		eigenPrincip.append(vecPrincip)             # Obtention de vecteurs unitaires 1*22500
	print("Vecteurs principaux crées et normalisés")

	inertieCumul = 0
	index = -1
	kEigenPrincip = []
	while inertieCumul < np.sum(eigFacValPro)*0.8 :     # cherche le nombre de composantes à garder afin
		inertieCumul += eigFacValPro[index]             # de garder une inertie de 80%
		kEigenPrincip.append(eigenPrincip[index])       # on insère les k vecteurs principaux à garder
		index -= 1
	k = len(kEigenPrincip)
	kEigenMat = np.array([np.array(xi) for xi in kEigenPrincip], dtype="object")    # on transforme ces vecteurs en une matrice
	poids = np.dot(galVY, kEigenMat.T)  # obtention des poids dans une matrice n * k
	poidsL = poids.tolist()
	poidsList = []
	for i in range(0,len(poidsL)):
		poidsList.append(np.array(poidsL[i]))
	imgaTrouv = np.array(image.flatten())   # image requête linéarisé
	imgVMoy = imgaTrouv - moyenne
	poidImg = np.dot(imgVMoy,kEigenMat.T)   #

	return poidsList,poidImg

"""plt.figure()
	plt.title("Carte des individus")
	plt.xlim(c1.min()-5, c1.max()+5)
	plt.ylim(c2.min()-5, c2.max()+5)
	plt.axis("on")
	plt.scatter(c1, c2, s=10, c='red', marker='+')
	plt.show()"""