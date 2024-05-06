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
bazel build parclusterer_exp/benchmark:parhac_main
```

```bash
# On iris with a single batch, static HAC
bazel run benchmark:run_experiment -- \
--input_data=$EXP_ROOT/iris/iris.scale.permuted.fvecs \
--ground_truth=$EXP_ROOT/iris/iris.scale.permuted_label.bin \
--clustering=$EXP_ROOT/result/parhac/iris \
--output_file=$EXP_ROOT/results/logs_iris \
--num_batch=1 --weight=0.3  \
--method=parhac \
--output_knn=$EXP_ROOT/result/knn/iris \
```

# GraphGrove
```
mkdir results/results_grove
```