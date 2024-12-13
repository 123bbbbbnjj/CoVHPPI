<br/>
<h1 align="center">Prediction and Evaluation of Coronavirus and Human Protein-Protein Interactions</h1>
<br/>

<br/>
<h2>Overview</h2>

The high lethality and infectiousness of coronaviruses, particularly SARS-Cov-2, pose a significant threat to human society. Understanding coronaviruses is crucial for mitigating the coronavirus pandemic. In this study, we conducted a comprehensive comparison and evaluation of five prevalent computational methods: interolog mapping, domain-domain interaction methodology, domain-motif interaction methodology, structure-based approaches, and machine learning techniques. These methods were assessed using unbiased datasets that include C1, C2h, C2v, and C3 test sets. Ultimately, we integrated these five methodologies into a unified model for predicting protein-protein interactions (PPIs) between coronaviruses and human proteins. Based on this model, we further established a high-confidence PPI network between coronaviruses and humans, consisting of 18,012 interactions between 3,843 human proteins and 129 coronavirus proteins. The reliability of our predictions was further validated through current knowledge framework and network analysis. 
<br/>
<img src="https://github.com/covhppilab/image/blob/main/Fig2.jpg"/>



Table of Contents
=================
* [ ⌛️&nbsp; Software Environment](#Environment)
* [ ⚗️&nbsp; Model download](#Model_download)
* [ 🚀&nbsp;Datasets](#Datasets)
* [ ⌛️&nbsp; Five Computational-Based Methods for PPI Prediction](#Five)
  * [ ⌛️&nbsp; Machine Learning Techniques (ML)](#(ML))
  * [ 🧬&nbsp;  Interolog Mapping (IM)](#(IM))
  * [ 💥&nbsp;  Domain-Domain Interaction Methodology (DDI)](#(DDI))
  * [ 🧠&nbsp; Domain-Motif Interaction Methodology (DMI)](#(DMI))
  * [ ⚗️&nbsp; Structure-Based Approaches (S) ](#(S))
* [ 🧐&nbsp;  Integrated Model ](#Integrated_Model)
* [ 📈&nbsp; Predicted ](#Predicted)
* [ ❤️&nbsp;Contact ](#Contact)



<a name="Environment"></a>
## ⌛️&nbsp; Software Environment
```
conda env create -f environment.yaml
conda activate cov
```


<a name="Model_download"></a>
## ⚗️&nbsp; Model download

1. EsmMean (2560)
Description: EsmMean encoding is the average length of ESM 2 in the protein output vector of the final layer. Note: The ESM 2 used here is the pre-trained model esm2_t36_3B_UR50D.
Model Code: [facebookresearch/esm](https://huggingface.co/facebook/esm2_t36_3B_UR50D )
2. ProtTrans (1024)
Description: ProtTrans uses UniRef and BFD (Big Fantastic Database) datasets as corpora and employs autoregressive and autoencoder models to generate protein representations. The pre-trained model used here is prot_t5-xx_half_unref50 enc. For each protein, the ProtT5 model is used to generate a final representation of L × 1024, where L is the length of the protein. In this work, ProtTrans encoding is the length-averaged vector of the output (L × 1024) from ProtT5.
Code Reference: [agemagician/ProtTrans](https://github.com/agemagician/ProtTrans )
3. Doc2vec (32)
Description: In an unsupervised doc2vec embedding learning framework, the feature representation of continuous protein sequences is based on the assumption that a set of protein sequences forms a "document".
Code Reference: [Doc2vec](http://zzdlab.com/intersppi/hvppi/download/HVPPI.tar.gz)

<a name="Datasets"></a>
## 🚀&nbsp; Datasets    
### Introduction

- **C1**: C1 represents the regular randomized partition of the test set; 
- **C2v and C2h**: C2v and C2h due to the involvement of two different species, virus and human
- **C3**: C3 represents the fact that none of the proteins in the test set can be found in the training set. 
- **Fold5**:five-fold cross-validation comparisons

### Dataset    
- **Bulid C1C2C3**:
  -  `python data/build_C1C2C3.py`
- **Bulid Fold5**:
  -  `python data/build_Fold5.py`

<a name="Five"></a>
## ⌛️&nbsp; Five Computational-Based Methods for PPI Prediction    
Within our computational framework, each pair of proteins is provided as input in the form of sequences or structures, and the predicted probability (Pr) of each method is output through five prevalent computational-based methods (ML, IM, DDI, DMI, and S). Finally, five different prediction probabilities were integrated adopting the Stacking strategy with Random Forest (RF) to derive the final interaction score.
<a name="(ML)"></a>
## ⌛️&nbsp; Machine Learning Techniques (ML)
  
- **Feature Extraction**:
  - Amino acid composition (AAC) and order:
    ```    
    - `python ML/features/CKSAAP.py`
    ```
  - Evolutionary information:
    ``` 
    - `python ML/features/get_pssm.py`
    ```
  - Protein embeddings:
    ```
    - `python ML/features/doc2vec.py`
    - `python EMS2.py  esm2_t36_3B_UR50D  ../../data/v_and_h.fasta tesm_out/ --include mean --repr_layers 36 --truncation_seq_length 4000 --save_file v_and_h_esm2.pkl `
    - `python ML/features/prottrans.py`
    ```
  - other features
    ```
    - `Rscript feature.R`
    ```
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
```
  - `python 10folds_RF_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean`
  - `python 10folds_AB_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean`
  - `python 10folds_NN_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean`
  - `python 10folds_SVM_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean`
  - `python 10folds_XGB_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean`
```


<a name="(IM)"></a>
## 🧬&nbsp;  Interolog Mapping (IM)

- The IM method is mainly based on the homology of protein sequences for inference .
The quality of each PPI template was evaluated with the HIPPIE strategy , which assigns a confidence score to each PPI template (SIM). To further identify homology of query protein pairs between viruses and humans, BLAST was used to search for identifying their homologues (E-value≤10<sup>-5</sup>, sequence identity≥30%, and alignment coverage of query protein≥40%).

- Obtain the training set of C1C2hC2vC3
  ```  
  - `python IM/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py`
  ```
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
  ```    
  - `python IM/C1C2C3/10folds_C1223_IM.py`
  ```


<a name="(DDI)"></a>
## 💥&nbsp;  Domain-Domain Interaction Methodology (DDI)

- The DDI method forecasts how likely two proteins will interact by looking at known domain-domain interactions. It starts with a DDI template library made using a strategy similar to HVIDB, where each template gets a confidence score. For predicting coronavirus-human interactions, positive samples proteins are checked with Hmmscan for Pfam domains (E-value ≤ 10<sup>-5</sup>). Unique domain pairs for coronavirus and human proteins are identified. These pairs confidence scores are based on how often they occur, scaled to a 0-1 range. The final scores (SDDI) are the average confidence scores of these domain pairs.

-Obtain the training set of C1C2hC2vC3    
  ```
  -  `python DDI/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py`
  ```
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
  ```
  -  `python DDI/C1C2C3/10folds_C1223_DDI.py`
  ```



<a name="(DMI)"></a>
## 🧠&nbsp; Domain-Motif Interaction Methodology (DMI)

- The DMI method predicts the likelihood of interactions between human proteins and viral motifs. It's seen as the most biologically relevant approach . DMI data was sourced from the 3did database to form a template library. The traditional DMI method uses this data for PPI prediction, so all DMI templates were given a default confidence score of 0.5. Like the DDI method, human protein domains were identified with Hmmscan, and viral protein motifs were found using regular expressions in the positive samples. Frequent domain-motif pairs were filtered out to get specific human-coronavirus pairs. The confidence score for each domain-motif pair template was normalized to a 0-1 scale based on their frequency of occurrence. The domain and motif pairs dataset specific to coronavirus and humans was combined with the DMI template library, and the average confidence scores of the domain and motif pairs were calculated as the final scores (SDMI).

- Obtain the training set of C1C2hC2vC3
  ``` 
  - `python DMI/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py`
  ```
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
  ```   
  - `python DMI/C1C2C3/10folds_C1223_DMI.py`   
  ```

<a name="(S)"></a>
## ⚗️&nbsp; Structure-Based Approaches (S)

- Compared to sequences, there is a lack of data for structures, although the relative conservatism of structures favors the prediction of PPIs. Similar to the IM method, the S method makes inferences based on the similarity of protein structures. The template dataset used here is the structural data corresponding to the template dataset in the IM method, and is also given the same confidence score (SS) as the IM method. The structural data was obtained from the PDB [58]. Structural simulations were performed using AlphFold2 for proteins without structural annotations.. The relatively advanced US-align algorithm [60] was used to compare the protein structures between virus and virus as well as between virus and human, with TM-score≥0.5 as the judgment criterion for the structural similarity of two proteins.    
-Obtain the training set of C1C2hC2vC3
  ```
  - `python S/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py`
  ```
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
  ```
  - `python S/C1C2C3/10folds_C1223_S.py`   
  ```


<a name=" Integrated_Model"></a>
## 🧐&nbsp;  Integrated Model
- For a more comprehensive prediction of coronaviruses and human PPIs, we finally constructed an integrated model with RF as a meta learner. This model synthesizes five methods to further improve the performance in the C2v and C3 test sets. Additionally, three other widely-used meta learners were chosen for benchmarking purposes. The result is shown in Fig. 8 and Supplementary Table 11, where CK indicates the ML method exhibits relatively superior performance among the five prediction methods.
In order to comprehensively predict the interaction between coronaviruses and human proteins, we ultimately constructed an ensemble model using RF as a meta learner by integrating the output probabilities of five methods.
  ```
  - `python Integrated_model/C1C2C3_RF.py`
  - `python Integrated_model/C1C2C3_LR.py`
  - `python Integrated_model/C1C2C3_NB.py`
  - `python Integrated_model/C1C2C3_SVM.py`
  ```
  
<a name="Predicted"></a>
## 📈&nbsp; Predicted
**Threshold_setting**:
```
  - `python final_predict/threshold_setting.py`
```   
**Run the Model**:       
```
  - `python final_predict/predict.py`
```
<a name="Contact"></a>
## ❤️&nbsp; Contact
For any questions or collaborations, please contact Jia Wang at wang.jia@mail.hzau.edu.cn.
