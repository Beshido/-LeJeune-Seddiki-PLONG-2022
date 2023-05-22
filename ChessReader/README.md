```bash
python3 -m venv env # crée un environnement virtuel où installer les dépendances du programme
source env/bin/activate # active l'environnement virtuel crée
pip install -r requirements.txt # télécharge et installe les dépendances de cet environnement
```

# Lancer les tests unitaires
```bash
python3 -m unittest -v
```

# Lancer la documentation
```bash
python -m pydoc -p {port}
```

# Version de Python
Python 3.5, 3.6 ou 3.7 pour que Tensorflow fonctionne avec DirectML.

## Liens intéressants
* [Convert a physicial chessboard into a digital one](https://tech.bakkenbaeck.com/post/chessvision)
* [Neural Chessboard](https://github.com/maciejczyzewski/neural-chessboard)
* [chesscog](https://github.com/georg-wolflein/chesscog)