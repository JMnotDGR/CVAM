# CVAM

## Introduction
CVAM is a tool to infer the CNV profile from spatial transcriptome data based on variational graph autoencoder (VGAE) and HMM under multi-level filtering.

## Applications and analysis of CVAM
### 1. Data preparation

EXPRESSION_PATH: the gene expression matrix of the spatial transcriptome data which we need to analyze. The row is Ensemble ID and the column is the spot name.

REFERGENEEXP: the reference gene expression matrix of the normal samples.

PQ_PATH: the cytoband information.

REFERENCE_PATH: the reference genome information.

```python
EXPRESSION_PATH = "gene_expression.tsv"
REFERGENEEXP = "reference_expression.tsv"
PQ_PATH = "hg38cytoBand.txt"
REFERENCE_PATH = "hg38refer.txt"
```
### 2. Create CVAM object
Create CVAM object. The ST_flag represent whether the data belong to spatial transcriptome. If false, the input is not spatial transcriptome data. The data will be map to a 2-D space so that the form of data is the same with spatial transcriptome data through novoSpaRc.

```bash
obj = tool.CVAM(EXPRESSION_PATH, PQ_PATH, REFERENCE_PATH, RES_PATH, ST_flag=True)
```
### 3. Cluster and CNVs inference
Cluster the data and infer the CNVs. The Model_number is the mode iterate times. m_out and the dim_output is the number of node in the hidden layers. LEARNING_RAT is the learnng rate in the network.
```bash
predict_label = obj.cluster(Model_number, m_out, dim_output, LEARNING_RAT)
res = obj.inferCNV(REFERGENEEXP, predict_label.values)

# Visual clustering results
classify = predict_label.sort_values("cluster")
col = classify["location"].values
refer = pd.read_csv("hg38refer.txt", sep='\t', header=None)
refer.columns = ["id", "chr", "start", "end", "name"]
cnv = res[col]
location = model.getLocation(col)
draw.zplot(location, predict_label)
plt.show()

# Visual CNV profile
refer.index = refer["id"]
cnv = cnv.join(refer, how='inner')
cnv.index = cnv["name"].values
cnv = cnv[col]
draw.drawprofile(cnv, refer, classify)
plt.show()
```
### 4. co-occurrence and mutual exclusion analysis

The co-occurring and mutual-exclusive CNV event can be analyzed through the one-sided Fisher’s test. k is the cluster id we need to analyze.

```bash
k = 1
classify2 = copy.deepcopy(classify.loc[classify["cluster"] == k, :])
col2 = classify2["location"].values
cnv2 = copy.deepcopy(cnv[col2])
cnv2 = cnv2[~cnv2.index.duplicated()]
all_result, occ, mut = analyze.mutualstate(cnv2)
#Visual results
draw.mutAndOcc(occ, mut)
plt.show()
```

### 5.  CNV multi-distance spatial pattern analysis

The spatial point pattern of the CNV event from multi-distance is analyzed by Ripley's K-function.

rrange: the set of distance we want to compute.

path: the path we store the results.

```bash
#pattern
rrange = [1,2,5]
path = "pattern"
analyze.inferPattern(cnv, rrange, path)
```


