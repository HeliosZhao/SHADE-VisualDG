
## test
OUT_DIR=$1
CHECK_ID=$2


DATASET=duskrainy
python test.py --dataset_test voc_2007_train_${DATASET} --checkepoch ${CHECK_ID} --cuda --output_dir ${OUT_DIR}

DATASET=nightrainy
python test.py --dataset_test voc_2007_train_${DATASET} --checkepoch ${CHECK_ID} --cuda --output_dir ${OUT_DIR}

DATASET=daytimefoggy
python test.py --dataset_test voc_2007_train_${DATASET} --checkepoch ${CHECK_ID} --cuda --output_dir ${OUT_DIR}

DATASET=nightclear
## 
python test.py --dataset_test voc_2007_train_${DATASET} --checkepoch ${CHECK_ID} --cuda --output_dir ${OUT_DIR}
