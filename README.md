Créer un environnement virtuel où installer les dépendances du programme : ```python3 -m venv env```

Activer l'environnement virtuel crée : ```source env/bin/activate```

Télécharger et installer les dépendances dans cet environnement : ```pip install -r requirements.txt```

Utiliser les commandes GUI de OpenCV via WSL2 avec VcXsrv : ```export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0 && export LIBGL_ALWAYS_INDIRECT=1```

## Liens intéressants
* [Convert a physicial chessboard into a digital one](https://tech.bakkenbaeck.com/post/chessvision)
* [Neural Chessboard](https://github.com/maciejczyzewski/neural-chessboard)