# Dynamic Hierarchical Agglomerative Clustering


## Generate Data
```bash
mkdir data
mkdir results
export EXP_ROOT=$(pwd) # set EXP_ROOT to current directory
# Iris
python3 utils/fvecs_converter.py --data_file=$EXP_ROOT --output_file=$EXP_ROOT/data/iris/iris.scale.permuted --data=iris
# MNIST
python3 utils/embed_mnist.py --output=$EXP_ROOT/data/mnist
python3 utils/fvecs_converter.py --data_file=$EXP_ROOT/data/mnist --output_file=$EXP_ROOT/data/mnist/mnist.scale.permuted --data=mnist
# ALOI 
# download data
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/aloi.scale.bz2
bzcat aloi.scale.bz2 > aloi.scale.raw
python3 utils/fvecs_converter.py --data_file=$EXP_ROOT/aloi.scale.raw --output_file=$EXP_ROOT/data/aloi/aloi.scale.permuted --data=aloi
# remove raw data
rm aloi.scale.raw
rm aloi.scale.bz2

# Imagenet
python3 utils/fvecs_converter.py --data_file=$EXP_ROOT/data/ilsvrc_small/ilsvrc_small.npy --output_file=$EXP_ROOT/data/ilsvrc_small/ilsvrc_small.scale.permuted --data=ilsvrc_small
```

## Run experiments
```bash
bazel build parclusterer_exp/benchmark:cut_dendrogram
bazel build parclusterer_exp/benchmark:parhac_main
```

```bash
export PARLAY_NUM_THREADS=1
# On iris with a single batch, static HAC
python3 parclusterer_exp/benchmark/run_experiment.py \
--input_data=$EXP_ROOT/data/iris/iris.scale.permuted.fvecs \
--ground_truth=$EXP_ROOT/data/iris/iris.scale.permuted_label.bin \
--clustering=$EXP_ROOT/results/parhac/iris \
--output_file=$EXP_ROOT/results/logs_iris \
--num_batch=1 --weight=0.3  \
--method=parhac \
--first_batch_ratio=1 \
--output_knn=$EXP_ROOT/results/knn/iris \
--store_batch_size=1

./parclusterer_exp/run_hac.sh 
./parclusterer_exp/run_dynhac_full.sh 
./parclusterer_exp/run_dynhac.sh 
./parclusterer_exp/run_deletion.sh 
```

# GRINCH
```bash
./parclusterer_exp/run_grinch.sh mnist 
./parclusterer_exp/run_grinch.sh aloi
./parclusterer_exp/run_grinch.sh ilsvrc_small
```

# GraphGrove

Install from source first: https://github.com/nmonath/graphgrove/tree/main. Use python version compatible with cut_dendrogram. >=3.9. e.g. `conda install python=3.12`.

```bash
conda activate gg
conda install --yes --file requirements.txt
bazel build //parclusterer_exp/benchmark:cut_dendrogram
mkdir results/results_grove
python3 parclusterer_exp/benchmark/grove_main.py --dataset=mnist
python3 parclusterer_exp/benchmark/grove_main.py --dataset=aloi
python3 parclusterer_exp/benchmark/grove_main.py --dataset=ilsvrc_small
```



# instructions for imagenet data
```bash
pip install kaggle
kaggle competitions download -c imagenet-object-localization-challenge
curl https://sh.rustup.rs -sSf | sh
. "$HOME/.cargo/env" 
sudo apt-get install pkg-config openssl libssl-dev
cargo install ripunzip
```

install [miniconda](https://docs.anaconda.com/free/miniconda/)

```bash
conda create -n "py37" python=3.7 
conda activate py37
conda install -c conda-forge tensorflow=1.14
conda install -c conda-forge six
conda install --yes --file requirements.txt

# sample 50K images from 1000 classes
./utils/sample_images.sh 
# double check there are 50K images
find /home/sy/imagenet/images/ -type d -maxdepth 1 | wc -l
# embed
python3 utils/embed_imagenet.py /home/sy/imagenet/images /home/sy/imagenet/
```



## Plotting

```bash
mkdir plots
python3 plot/plot_main.py
```