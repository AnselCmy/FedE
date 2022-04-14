# Experimental Data

We put datasets **FB15k-237-Fed3**, **FB15k-237-Fed5**, **FB15k-237-Fed10**, and **NELL-995-Fed3** here.

## Load a dataset
You can load a dataset using the package pickle:

```python
import pickle
all_data = pickle.load(open('./data/FB15k237-Fed3.pkl', 'rb'))
```

## Arrangement of datasets
Each dataset is a ___list___ of ___dict___, specifically, each ___dict___ saves the KG data for one client in a dataset. For example, as the data loaded from above code block, ```all_data[0]``` holds the KG data in clinet 1 and ```all_data[1]``` holds the KG data in clinet 2 of FB15k237-Fed3.

In the each ___dict___, for instance ```all_data[0]```, its structure is like:

```
{
'train':{
		  'edge_index': array([[...], [...]]),
		  'edge_type': array([...]),
   		  'edge_index_ori': array([[...], [...]]),
		  'edge_type_ori': array([...])
		  }
'valid':{
		  'edge_index': array([[...], [...]]),
		  'edge_type': array([...]),
   		  'edge_index_ori': array([[...], [...]]),
		  'edge_type_ori': array([...])
		  }
'test':{
		  'edge_index': array([[...], [...]]),
		  'edge_type': array([...]),
   		  'edge_index_ori': array([[...], [...]]),
		  'edge_type_ori': array([...])
		  }
}
```
In this ___dict___ for saving a KG of one client, the value of ```train```, ```valid```, and ```test``` specify the KG for model training, validation and test. ```edge_index``` shows the head and tail indices of triples, and ```edge_type``` show the relation indices of triples. Moreover, ```edge_index_ori``` and ```edge_type_ori``` show the original indeces of head, tail, and relations from original datasets. The original **FB15k-237** and **NELL-995** are taken from [OpenKE](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/benchmarks).