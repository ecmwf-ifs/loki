
##  Using the notebooks

The notebooks contain many `import`  statements for loading various Loki modules. In order to ensure the interpreter can locate the relevant modules, the jupyter notebook server should be launched from a terminal where the `loki-env` virtual environment has been activated:

```text
$ source loki-activate
(loki_env) $ jupyter notebook
```

The interpreter used by the notebook server can be checked using the following command:

```text
(loki_env) $ jupyter kernelspec list
Available kernels:
  python3    <path-to-loki>/loki_env/share/jupyter/kernels/python3
```
