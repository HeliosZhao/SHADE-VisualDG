PY_ARGS=${@:1}

python -W ignore train_shade.py --SHM --sets a-all ${PY_ARGS}

python -W ignore train_shade.py --SHM --sets c-all ${PY_ARGS}

python -W ignore train_shade.py --SHM --sets p-all ${PY_ARGS}

python -W ignore train_shade.py --SHM --sets s-all ${PY_ARGS}