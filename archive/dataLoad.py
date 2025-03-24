import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import img

gallery = []
probes = []
groundtruth = []

"""
Le but de notre fonction va être de charger les différentes images d'individus en trois listes:
	- gallery
	- probes
	- groundTruth
Les images dans probes vont varier en fonction du paramètre n, permettant de choisir n images
par individus dans les 100 premiers élements de cette liste
"""
def dataLoad(n):
	path = "D:/louis/Documents/Institut Mines-Télécom Lille-Douai 2017-2023/4-M1 - 2020-2021/Modules/P2 - ODATA - Outils pour la Data Science/Projet Authentification faciale/data/dataset2/images"
	print("Chargement des données en cours... patientez")
	probes1 = []
	probes2 = []
	maxprobes = 100
	currentName = ''
	compteur = 0

	for root, _, files in os.walk(path):        #mise en place du routage dans le dossier dataset
		current_directory_path = os.path.abspath(root)
		for f in files:                                # parcous de chaque fichier du dossier courant
			nameVersion, ext = os.path.splitext(f)
			if ext == ".jpg":                          # vérification qu'il sagit bien d'un fichier .jpg
				current_image_path = os.path.join(current_directory_path, f)
				current_image = mpimg.imread(current_image_path)        # création de la matrice de l'image (150,150)
				name, version = os.path.splitext(nameVersion)
				groImgT = img.image(name,version,current_image,True)        # création de deux variables classe image
				groImgF = img.image(name, version, current_image, False)    # nom, version, matrice et savedGallery

				if len(probes2) < maxprobes:        # nous cherchons à remplir la liste probes2
					probes2.append(current_image)
					groundtruth.append(groImgF)
				else:
					if len(probes1) < maxprobes:         # nous cherchons aussi à remplir la liste probes1
						if name == currentName:          # avec n photos de chaque individu après probes2
							if compteur < n:
								probes1.append(current_image)
								groundtruth.append(groImgF)
								compteur += 1
							else:                               # si on a déjà n photos de tel individu
								gallery.append(current_image)
								groundtruth.append(groImgT)
						else:                                   # une fois que l'on change d'individu
							compteur = 1
							probes1.append(current_image)
							groundtruth.append(groImgF)
							currentName = name                  # on associe au nom stocké le nom du nouvel individu
					else:                                       # une fois que les deux probes sont pleins
						gallery.append(current_image)
						groundtruth.append(groImgT)

	for l in probes1:       # fusion des deux lists en probes
		probes.append(l)
	for l in probes2:
		probes.append(l)

	print("Gallery : ", len(gallery), "\tProbes : ", len(probes), "\tGroundtruth : ", len(groundtruth))
	print("Chargement des images: complété")