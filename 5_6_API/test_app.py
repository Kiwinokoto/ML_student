# fichier pytest

# Avec flask, la partie du code "avant les routes" est + difficile à débuguer
# (dans les routes, on peut facilement renvoyer une erreur. En dehors,
# on peut juste savoir que le site ne s'affiche pas).

# Au début on peut simplement mettre tout le code dans une route, mais lors du déploiement
# il faut optimiser, sinon le temps de réponse est impacté.
# C'est donc une partie du code où les tests unitaires peuvent être particulièrement importants / utiles.

# Ici nous allons vérifier que la fonction preprocess_text met bien en forme comme attendu
# une chaine de characteres (= on vérifie que l'input pour notre modèle a le bon format).
# On va aussi s'assurer que le texte est bien en minuscules, et que les stopwords ont bien été filtrés.


# Import test function from app.py
from app import preprocess_text


with open('./forbidden_words.txt', 'r') as file:
        forbidden = [line.strip() for line in file]


def test_preprocessing():
    test_query = "A noRmal strinG of chaRACTERS, about python and javascript and stuff."
    result = preprocess_text(test_query)
    filtered_result = [word for word in result if word not in forbidden]

    # check result is a list
    assert isinstance(result, list)

    # Check result contains only strings
    assert all(isinstance(word, str) for word in result)

    # no uppercase
    assert result == [x.lower() for x in result]

    # no stopwords
    assert result == filtered_result

