res_dir="./reports/"
model="Spotscape"
trials=10
mb="False"
e=1000
ee=100

# Dataset
d="10X_DLPFC"
p=0
s=0

# Parameters
lr=0.0001

python main.py \
    --embedder $model --result_dir $res_dir --device $dev \
    --use_minibatch $mb --epochs $e --ee $ee --trials $trials \
    --dataset $d --patient_idx $p --slice_idx $s --lr $lr

