import dataLoad as load
import authentification as auth
import numpy as np
import time

load.dataLoad(3)

gallery = load.gallery
probes = load.probes
groundTruth = load.groundtruth

"""
:param :    image:      matrice de l'image

:objectifs: renvoie l'identité de l'image et vérifier l'utilisateur est
			enregistré dans la gallery

:returns:   indivEnregistre :   booleen pour savoir s'il est dans la gallery
			identity:           nom de l'individu
			index:              index des autres images de l'utilisateur dans groundtruth
			
"""
def evaluate(image):
	identity = ''
	index = []

	indivEnregistre = False

	for l in groundTruth:
		comparaison = l.matrix == image
		if comparaison.all():
			identity = l.name
			break
	i = 0
	if identity != '':
		for l in groundTruth:
			if identity == l.name:
				index.append(i)
				if l.savedGal == True:
					indivEnregistre = True
			i += 1
	return indivEnregistre, identity, index

# TP - Vrai positif : cas 1 : autorise à un utilisateur justement enregistré
# FP - Faux positif : cas 2 : autorise à utilisateur enregistré, mais par mauvaise reconaissance
# FP - Faux positif : cas 3 : autorise à utilisateur non-enregistré, mais par reconaissance proche
# TN - Vrai négatif : cas 4 : refuse à un utilisateur non-enregistré
# FN - Faux négatif : cas 5 : refuse à un utilisateur justement enregistré
"""
:param :    image:      matrice de l'image
			dataset:    liste de matrices d'images
			radius:     distance minimum pour les voisins proches
			
:objectifs: Cette fonction a pour objectif d'analyser l'image et en déduire dans quel cas
			l'image se situe. Pour l'analyse, la fonction va comparer les matrices d'images de ground-truth
			et des voisins proches

:returns:   array([TP, FP, FN, TN]) : renvoie les mesures d'évaluation de l'image par rapport aux différents cas

"""
def eMesure(image, dataset, radius):
	indivEnregistre, identite, index = evaluate(image)
	voisinsProches = auth.bfs.radius_search(dataset, image, radius)[0]
	authentifie = auth.bfAuth(image, dataset, radius)

	TP = 0
	FP = 0
	FN = 0
	TN = 0

	voisinsValides = []

	if indivEnregistre == True:
		for i in range(0, len(voisinsProches)):
			for j in range(0, len(index)):
				comparaison = dataset[voisinsProches[i]] == groundTruth[index[j]].matrix
				if comparaison.all():
					voisinsValides.append(voisinsProches[i])
					break
				else:
					continue

		if len(voisinsProches) == len(voisinsValides):
			image_reconnue = True
		else:
			image_reconnue = False

	if authentifie == True:
		if indivEnregistre == True:
			if image_reconnue == True:  # cas 1:    True Positive
				TP = 1
			else:                       # cas 2:    False Positive2
				FP = 1
		else:                           # cas 3:    False Positive3
			FP = 1
	else:
		if indivEnregistre == True:     # cas 5:    False Negative
			FN = 1
		else:                           # cas 4:    True Negative
			TN = 1
	return np.array([TP, FP, FN, TN])


"""
:param :    TP: True Positive
			FP: False Positive
			FN: False Negative
			TN: True Negative
			
:objectifs: Cette fonction a pour objectif d'évaluer nos mesures en fonction des paramètres

:returns:   exactitude:
			precision:  
			sensibilite:
			specificite:

"""
def evaluationMes(TP,FP,FN,TN):
	exNum = TP+TN
	exDen = TP + FP + FN + TN
	if exDen != 0:
		exactitude = exNum/exDen
	else:
		exactitude = 0

	preDen = TP + FP
	if preDen != 0:
		precision = TP / preDen
	else:
		precision = 0

	senDen = TP + FN
	if senDen != 0:
		sensibilite = TP / senDen
	else:
		sensibilite = 0

	speDen = FP + TN
	if speDen != 0:
		specificite = TN / speDen
	else:
		specificite = 0

	return exactitude, precision, sensibilite, specificite


def casSuccErr(radius):
	bfCasP1 = np.zeros(4)

	print("Calcul des Mesures en cours...")
	for i in range(0, 100):
		bfCasP1 += eMesure(probes[i], gallery, radius)
		bfCasP1 += eMesure(probes[i+100], gallery, radius)

	excatP1,precisP1,sensiP1,specifP1 = evaluationMes(bfCasP1[0],bfCasP1[1],bfCasP1[2],bfCasP1[3])
	print("Evaluation pour un brute force search pour Probes avec radius = ", radius)
	print("Exactitudes : \tP1 ", excatP1)
	print("Precision : \tP1 ", precisP1)
	print("Sensibilite : \tP1 ", sensiP1)
	print("Specification :\tP1 ", specifP1)

	return bfCasP1

start_time = time.time()
bf1G = casSuccErr(1000000)
interval = time.time() - start_time
print 'Total time in seconds:', interval
bf1H = casSuccErr(1700000)
bf1I = casSuccErr(1800000)
bf1J = casSuccErr(1900000)
bf1K = casSuccErr(2000000)
