# CVAM

## Introduction
CVAM is a tool to infer the CNV profile from spatial transcriptome data based on variational graph autoencoder (VGAE) and HMM under multi-level filtering.

## Applications and analysis of CVAM
### 1. Data preparation

The content that users need to input mainly includes five parts:

- EXPRESSION_PATH: the gene expression matrix of the spatial transcriptome data which we need to analyze. The row is Ensemble ID and the column is the spot name.



- REFERGENEEXP: the reference gene expression matrix of the normal samples.



- PQ_PATH: the cytoband information about the p/q location of each chromosome.



- REFERENCE_PATH: the reference genome information.
- RES_PATH: the path (only a string) for reserved results.

*Please refer to the "reference" folder in GItHub for examples of the four part formats

```python
EXPRESSION_PATH = "example_expression.tsv"
REFERGENEEXP = "example_reference_expression.tsv"
PQ_PATH = "example_cytoBand.txt"
REFERENCE_PATH = "example_refer.txt"
```
### 2. Create CVAM object
Create CVAM object. alpha represents the filtering percent, which filters out genes expressed below alpha. sigma represents the filtering standard deviation, which filters out genes expressed standard deviation below sigma. If you want to preserve as many genes as possible, you can set alpha and sigma smaller. ST_flag represent whether the data belong to spatial transcriptome. If false, the input is not spatial transcriptome data. The data will be map to a 2-D space so that the form of data is the same with spatial transcriptome data through novoSpaRc.

```python
obj = tool.CVAM(EXPRESSION_PATH, PQ_PATH, REFERENCE_PATH, RES_PATH, alpha=0.05, sigma=0.5, ST_flag=True)
```
### 3. Cluster and CNVs inference
Cluster the data and infer the CNVs. The Model_number is the mode iterate times which influence the model accuracy. The larger the value, the more sufficient the model training is. m_out and the dim_output is the number of node in the hidden layers. LEARNING_RAT is the learnng rate in the network. solution represents clustering resolution, smaller values result in more clusters. scale is the size of multi-level median filtering. The larger the value, the larger the smoothing scale, and the stronger the noise resistance of the model. windows is the init size of median filtering window. The larger the value, the more genes covered. delta is the smooth step so that the window size is different with different median-filtering.
```python
predict_label = obj.cluster(Model_number, m_out, dim_output, LEARNING_RAT, solution=2)
############expected result#####################
raw exp shape:  (31507, 1906)
model_data shape: (1906, 6721)
running VGAE
Time:  1
Loss: 35.494 AL: 0.702 XL: 1.442 KL: 34648.055
Loss: 166.794 AL: 0.554 XL: 1.188 KL: 166121.406
Loss: 126.723 AL: 0.517 XL: 1.212 KL: 126085.594
Loss: 93.077 AL: 0.415 XL: 1.000 KL: 92562.461
Loss: 71.253 AL: 0.321 XL: 0.890 KL: 70842.844
Time:  2
Loss: 58.913 AL: 0.259 XL: 0.823 KL: 58572.461
Loss: 50.288 AL: 0.202 XL: 0.785 KL: 50007.461
Loss: 44.854 AL: 0.159 XL: 0.740 KL: 44621.211
...
...
###############################################

res = obj.inferCNV(REFERGENEEXP, predict_label.values, scale=3, windows=31, delta=20)

# Visual clustering results
classify = pd.read_csv(RES_PATH + "/cluster.txt", sep='\t', header=None)
col = classify[0].values
location = model.getLocation(col)
predict_label = classify[1].values + 1
draw.zplot(location, predict_label, max(predict_label))
plt.show()
```
![fenbu](D:\CVAM\data\mouse_ST_lung cancer\fenbu.svg)

```bash
# Visual CNV profile
draw.drawprofile(res, refer, classify)
plt.show()
```

![cnv](D:\CVAM-origin\data\cnv.png)

### 4. co-occurrence and mutual exclusion analysis

The co-occurring and mutual-exclusive CNV event can be analyzed through the one-sided Fisher’s test. k is the cluster id we need to analyze.

```bash
k = 3
classify2 = classify.loc[classify["cluster"] == k, "location"]
all_result, occ, mut = analyze.mutualstate(res[classify2])
#Visual results
draw.mutAndOcc(occ, mut)
plt.show()
```

![occ and mut](D:\CVAM-origin\data\occ and mut.png)

### 5.  CNV multi-distance spatial pattern analysis

The spatial point pattern of the CNV event from multi-distance is analyzed by Ripley's K-function.

rrange: the set of distance we want to compute. The larger the value, the greater the space.

path: the path we store the results.

```bash
#pattern
rrange = [i for range(50, 160, 10)]
path = "pattern"
gene = "S100a8"
analyze.inferPattern(res.loc[gene], rrange, path)
```

![s100a8](D:\CVAM-origin\data\s100a8.png)
