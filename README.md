```bash
python3 -m venv env # crée un environnement virtuel où installer les dépendances du programme
source env/bin/activate # active l'environnement virtuel crée
pip install -r requirements.txt # télécharge et installe les dépendances de cet environnement
export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0
export LIBGL_ALWAYS_INDIRECT=1 # permet d'utiliser les commandes GUI de OpenCV via WSL2 avec VcXsrv
```

# Lancer les tests unitaires
```bash
python3 -m unittest -v
``` 

## Liens intéressants
* [Convert a physicial chessboard into a digital one](https://tech.bakkenbaeck.com/post/chessvision)
* [Neural Chessboard](https://github.com/maciejczyzewski/neural-chessboard)