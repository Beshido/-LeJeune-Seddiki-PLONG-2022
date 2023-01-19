Montrer des images avec Windows via WSL2 :
```export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0 && export LIBGL_ALWAYS_INDIRECT=1```

```pip install -r requirements.txt```