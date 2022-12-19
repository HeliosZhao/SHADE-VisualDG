PY_ARGS=${@:1}

python -W ignore train_shade_l2d.py --task PACS --SHM --sets a-all ${PY_ARGS}

python -W ignore train_shade_l2d.py --task PACS --SHM --sets c-all ${PY_ARGS}

python -W ignore train_shade_l2d.py --task PACS --SHM --sets p-all ${PY_ARGS}

python -W ignore train_shade_l2d.py --task PACS --SHM --sets s-all ${PY_ARGS}
