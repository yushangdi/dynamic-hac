# Dynamic Hierarchical Agglomerative Clustering

```
mkdir data
mkdir results
export EXP_ROOT=/home/ubuntu/dynamic-hac/data
# Iris
python3 fvecs_converter.py --data_file=$EXP_ROOT --output_file=$EXP_ROOT/iris/iris.scale.permuted --data=iris
# MNIST
python3 embed_mnist.py --output=/home/ubuntu/dynamic-hac/data
python3 fvecs_converter.py --data_file=/home/ubuntu/dynamic-hac/data --output_file=/home/ubuntu/dynamic-hac/data/mnist/mnist.scale.permuted --data=mnist
# ALOI 
# download data
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/aloi.scale.bz2
bzcat aloi.scale.bz2 > aloi.scale.raw
python3 fvecs_converter.py --data_file=$EXP_ROOT/aloi.scale.raw --output_file=benchmark/aloi/aloi.scale.permuted --data=aloi
# remove raw data
rm aloi.scale.raw
rm aloi.scale.bz2
```


# GraphGrove
```
mkdir results/results_grove
```