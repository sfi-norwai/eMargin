python pretrain.py eMargin harth -p configs/harthconfig.yml -s 1 --evaluate clustering
python pretrain.py eMargin sleepeeg -p configs/harthconfig.yml -s 2 --evaluate clustering
python pretrain.py eMargin sleepeeg -p configs/harthconfig.yml -s 3 --evaluate clustering

python pretrain.py eMargin sleepeeg -p configs/sleepconfig.yml -s 1 --evaluate clustering
python pretrain.py eMargin sleepeeg -p configs/sleepconfig.yml -s 2 --evaluate clustering
python pretrain.py eMargin sleepeeg -p configs/sleepconfig.yml -s 3 --evaluate clustering

python pretrain.py eMargin ecg -p configs/ecgconfig.yml -s 1 --evaluate clustering
python pretrain.py eMargin ecg -p configs/ecgconfig.yml -s 2 --evaluate clustering
python pretrain.py eMargin ecg -p configs/ecgconfig.yml -s 3 --evaluate clustering