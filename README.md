# Catégorisez automatiquement des questions Stack Overflow


***

Présentation rapide :

	Projet réalisé dans le cadre d'une formation OpenClassrooms :
	ML engineer (ingénieur en Machine Learning), projet 5.

	Notre objectif est simple, il s'agit de déployer en ligne un modèle prédictif. 
	Ce modèle doit être capable de proposer des tags pertinents pour des questions StackOverflow 
	(le corpus sur lequel il a été entrainé).

	Il reçoit donc des requêtes de type texte, et renvoit pour chacune une liste de tags.

***


Présentation + générale, points-clés :


	J'ai trouvé ce projet passionnant à réaliser, notamment parce qu'il est très complet :


- Il va de l'aquisition des données, depuis la db, à la mise en production automatisée du modèle via les github actions.

- Le preprocessing NLP est très intéressant, les outils modernes (pos_taggers par exemple) s'appuyant sur des modèles préentrainés (nltk, spacy, ...) 
offrent des possibilités de traitement et d'analyse efficaces, avec finesse. Malheureusement pour les codeurs, les regex restent très utiles :)

- A chaque projet j'ai découvert des outils prodigieusement utiles et performants (et souvent opensource. What a day to be alive!). Souvent, ces outils sont en fait d'une grande simplicité. Par exemple une PCA effectue simplement des projections, de manière optimale. Un kmeans définit des clusters aussi compacts que possible. Un dbscan permet élégamment de découvrir des structures imbriquées. Un knn copie sur les voisins... 
Sur ce projet, la NMF est un bon exemple d'outil qui a une certaine "évidence". On imagine très bien un mathématicien remarquer qu'une factorisation simple, 
quand elle est possible, pourrrait être interprêtée comme le découpage d'un corpus en thèmes (topics).
En revanche pour la LDA, je voudrais bien savoir qui a pensé à bricoler un truc pareil !
Il s'agit d'un algorithme de génération aléatoire de texte que l'on entraine à produire du texte similaire à ceux de notre corpus.
On observe ensuite les paramètres internes de la LDA pour décrire le corpus d'origine. A bit convoluted, isn't it ? C'est du génie, ou de la folie pure !

- En termes de machine learning, nous sommes ici sur un cas particulier très intéressant de classification multilabels, qui pose des problèmes spécifiques.
La mesure des scores est un point crucial. Le score utilisé doit être cohérent pour différents types de modèles (supervisés et non-supervisés), robuste à certaines modifications des hyperparamètres (comme le nombre de tags prédits), interprétable, compatible avec toutes les architectures utilisées (y compris tensorflow..) et les différents types d'outputs (en particulier quand il s'agit de distributions selon une loi multinomiale). Autrement dit, si la similarité Jaccard n'existait pas, il faudrait l'inventer !

- les méthodes .predict() natives de sklearn ne conviennent pas forcément ici, ce qui nous oblige (comme pour les scores) à définir des fonctions customs, voire des classes de modèles avec
pyfunct. Cela permet de comprendre dans le détail le fonctionnement des modèles.

- un côté sympa de l'approche mlops : elle donne l'impression de travailler en équipe, même quand on est seul sur un module ! 
Elle met en effet l'accent (entre autres) sur la coordination, le contexte global du projet sur le "life cycle" complet, et donc, les contraintes et besoins potentiels des équipes 
de collègues, en amont ou en aval du projet, ou développant d'autres modules.

- Au sujet du codage modulaire, j'aime le fait que ce projet y apporte une attention certaine. Même si l'utilisation de notebooks ici est plus pratique pour le partage et la visualisation
immédiate (lors de l'EDA par exemple) que pour une exécution automatisée du pipeline, l'architecture est bien modulaire : un notebook procède à la collecte des données.
Le suivant effectue le preprocessing NLP et exporte des fichiers .csv prêts à être train test splitted, embedded et utilisés par les modèles.
Les suivants permettent d'entrainer et d'évaluer différents modèles, en enregistrant les résultats de nos expériences grâce au tracking mlflow.
Ils intègrent la mise à jour du modèle en continue, par syncronisation via un worklow github actions. 

- Le déploiement automatisé de l'API constitue un véritable mini-projet en soi, les possibilités sont variées. J'ai choisi de tester d'abord l'utilisation d'un framework léger (flask) sur un hébergeur partagé (o2switch). C'est une solution qui fonctionne très bien, à l'exception des cas où la console de serveur s'est déconnectée. Le redémarrage de l'appli python peut alors prendre jusqu'à une minute (semble causé pour l'essentiel par la réactivation de l'environnement), avant l'affichage de la page web. Pour les prochains projets je privilégierai plutôt
des déploiements au sein de contenus dockers, beaucoup plus rapides dans ce cas (probablement sagemaker sur AWS).

- le point d'entrée (formulaire d'accès à l'API) est temporairement disponible à l'adresse https://kiwinokoto.com/ 


