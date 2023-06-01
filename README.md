# Polytech Lille - Projet ALN 2023

Code possible du sujet de projet d'analyse linéaire numérique de Polytech Lille pour les IS3 en 2023. Ce repo vise à aider les élèves des prochaines années en cas de difficultés sur un sujet similaire.

## Contenu

* Sujet original
* Rapport final au format pdf et tex dans le dossier rendus
* Fichier README
* Fonctions implémentées dans le dossier functions
* Fichiers de test

## Fonctions implémentées

* Hessenberg
* Gershgorin
* Bissection
* Séquence de Sturm
* Comptage changement de signe séquence de Sturm
* Puissance Inverse
* Solve Triangular
* Factorisation LU
* Eigh
* Bidiagonalisation
* Reflecteur
* SVD

Pour voir l’utilisation de ces fonctions sur le cas général de l’ACP appliqué
aux iris, utilisez les programmes suivants:
* ACP méthode historique : ```acp_iris_eigh.py```
* ACP méthode moderne : ```acp_iris_svd.py```

Pour tester chaque fonction indépendamment utilisez la commande :
```python3 test_unitaire.py <classe_de_la_fonction>```
