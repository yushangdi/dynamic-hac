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
kaggle competitions download -c imagenet-object-localization-challenge
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
```

# GraphGrove
```
mkdir results/results_grove
```