import dataLoad as load
import authentification as auth
import evalSys
import time

"""
:subject :  Projet d'authentification faciale en ODATA

:autors:    PIETTE Benoit, RIGAUX Louis
:date:      08/11/2020

:completed: False
"""
load.dataLoad(2)

gallery = load.gallery
probes = load.probes
groundTruth = load.groundtruth

def rechercheBruteForce(index, gallery, radius):
	start_time = time.time()
	if auth.bfAuth(probes[index],gallery,radius1) == True:
		print("Authentification validée : vous êtes autorisés")
	else:
		print("Authentification refusée : vous êtes autorisés")
	interval = time.time() - start_time
	print('Total time in seconds:', interval)

"""
Vous pouvez enlever la zone de commentaires pour accéder à la fonction et la tester
rechercheBruteForce(index,gallery,radius)
"""

print(auth.eigenAuth(probes[i],gallery,radius2))