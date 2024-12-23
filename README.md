<br/>
<h1 align="center">Prediction and Evaluation of Coronavirus and Human Protein-Protein Interactions</h1>
<br/>

<br/>
<h2>Overview</h2>

Coronaviruses, with SARS-CoV-2 being particularly deadly and contagious, pose a substantial threat to global health. Gaining insights into these viruses is essential for controlling the pandemic. In our study, we used unbiased datasets, including the C1, C2h, C2v, and C3 test sets, to compare and assess five computational methods: interolog mapping, domain-domain interaction methodology, domain-motif interaction methodology, structure-based approaches, and machine learning techniques. We then integrated these methods into a unified model, to predict protein-protein interactions (PPIs) between coronaviruses and human proteins.

<br/>
<img src="https://github.com/covhppilab/image/blob/main/fig.png"/>



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
  ```
  python data/build_C1C2C3.py
  ```
- **Bulid Fold5**:
  ```
  python data/build_Fold5.py
  ```
<a name="Five"></a>
## ⌛️&nbsp; Five Computational-Based Methods for PPI Prediction    

In our computational framework, each pair of proteins is sequence as a input. Machine learning (ML) technology converts these sequences into tensors that can be recognized by the model for training and testing. In addition to ML, we also employ four other computational methods: interolog mapping (IM), domain-domain interaction methodology (DDI), domain-motif interaction methodology (DMI), and structure-based approaches (S). These methods predict the probability of protein-protein interaction (Pr) by comparing the input information with the template database. Finally, we utilize the random forest (RF) algorithm to integrate the probabilities predicted by these five different methods, obtaining the final protein-protein interaction score.
<a name="(ML)"></a>
## ⌛️&nbsp; Machine Learning Techniques (ML)
- **Feature Extraction**:
<table>
    <tr>
        <th>Groups</th>
        <th>Encodings</th>
        <th>Dimension</th>
        <th>Code</th>
    </tr>
    <tr>
        <td rowspan="3">Amino acid composition and order</td>
        <td>AAC</td>
        <td>20</td>
        <td>R</td>
    </tr>
    <tr>
        <td>DC</td>
        <td>400</td>
        <td>R</td>
    </tr>
    <tr>
        <td>CKSAAP</td>
        <td>1200</td>
        <td>python ML/features/CKSAAP.py</td>
    </tr>
    <tr>
        <td rowspan="9">Amino acid physicochemical properties</td>
        <td>APseAAC</td>
        <td>80</td>
        <td>R</td>
    </tr>
    <tr>
        <td>PseAAC</td>
        <td>50</td>
        <td>R</td>
    </tr>
    <tr>
        <td>CT</td>
        <td>343</td>
        <td>R</td>
    </tr>
    <tr>
        <td>CTD</td>
        <td>147</td>
        <td>R</td>
    </tr>
    <tr>
        <td>QSO</td>
        <td>100</td>
        <td>R</td>
    </tr>
    <tr>
        <td>SOCN</td>
        <td>60</td>
        <td>R</td>
    </tr>
    <tr>
        <td>Geary</td>
        <td>240</td>
        <td>R</td>
    </tr>
   <tr>
        <td>Moran</td>
        <td>240</td>
        <td>R</td>
    </tr>
    <tr>
        <td>Moreau-Broto</td>
        <td>240</td>
        <td>R</td>
    </tr>
    <tr>
        <td rowspan="4">Evolutionary information</td>
        <td>AAC-PSSM</td>
        <td>20</td>
        <td>PSSM</td>
    </tr>
    <tr>
        <td>DPC-PSSM</td>
        <td>400</td>
        <td>PSSM</td>
    </tr>
    <tr>
        <td>RPSSM</td>
        <td>110</td>
        <td>PSSM</td>
    </tr>
    <tr>
        <td>PSSM-AC</td>
        <td>1200</td>
        <td>R</td>
    </tr>
 <tr>
        <td rowspan="3">Protein embedding</td>
        <td>EsmMean</td>
        <td>2560</td>
        <td>model</td>
    </tr>
    <tr>
        <td>ProtTrans</td>
        <td>1024</td>
        <td>model</td>
    </tr>
    <tr>
        <td>Doc2vec</td>
        <td>32</td>
        <td>model</td>
    </tr>
</table>  

NOTE:  <br> 
1.When the value of the code column is [R](https://github.com/nanxstats/protr):
  ```
  Rscript feature.R 
  ```
2.When the value of the code column is [PSSM](http://possum.erc.monash.edu/):<br>        
2.1Generate PSSM file:
   ```
   python ML/features/get_pssm.py
   ```     
2.2Generate features:
   ```
   python ML/features/PSSM_feature.py
   ```      
3.When the value of the code column is model:    <br>   
3.1Download model:<a href="#Model_download">Model_download</a>   <br>   
3.2Generate features: <br> 
   ```
   python ML/features/doc2vec.py     
   python EMS2.py esm2_t36_3B_UR50D ../../data/v_and_h.fasta tesm_out/ --include mean --repr_layers 36 --truncation_seq_length 4000 --save_file v_and_h_esm2.pkl     
   python ML/features/prottrans.py
   ```  







- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
```
  python 10folds_RF_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean
  python 10folds_AB_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean
  python 10folds_NN_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean
  python 10folds_SVM_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean
  python 10folds_XGB_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean
```


<a name="(IM)"></a>
## 🧬&nbsp;  Interolog Mapping (IM)

- The IM method is mainly based on the homology of protein sequences for inference .The quality of each PPI template was evaluated with the HIPPIE strategy , which assigns a confidence score to each PPI template (SIM). To further identify homology of query protein pairs between viruses and humans, [BLAST](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) was used to search for identifying their homologues (E-value≤10<sup>-5</sup>, sequence identity≥30%, and alignment coverage of query protein≥40%).

- Obtain the training set of C1C2hC2vC3
  ```  
  python IM/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py
  ```
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
  ```    
  python IM/C1C2C3/10folds_C1223_IM.py
  ```


<a name="(DDI)"></a>
## 💥&nbsp;  Domain-Domain Interaction Methodology (DDI)

- The DDI method forecasts how likely two proteins will interact by looking at known domain-domain interactions. It starts with a DDI template library made using the HVIDB strategy, where each template gets a confidence score. For predicting coronavirus-human interactions, positive samples proteins are checked with [Hmmscan](https://github.com/EddyRivasLab/hmmer) for Pfam domains (E-value ≤ 10<sup>-5</sup>). Unique domain pairs for coronavirus and human proteins are identified. These pairs confidence scores are based on how often they occur, scaled to a 0-1 range. The final scores (SDDI) are the average confidence scores of these domain pairs.

- Obtain the training set of C1C2hC2vC3    
  ```
  python DDI/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py
  ```
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
  ```
  python DDI/C1C2C3/10folds_C1223_DDI.py
  ```



<a name="(DMI)"></a>
## 🧠&nbsp; Domain-Motif Interaction Methodology (DMI)    

- The DMI method predicts the likelihood of interactions between human proteins and viral motifs. It's seen as the most biologically relevant approach . DMI data was sourced from the 3did database to form a template library. The traditional DMI method uses this data for PPI prediction, so all DMI templates were given a default confidence score of 0.5. Like the DDI method, human protein domains were identified with Hmmscan, and viral protein motifs were found using regular expressions in the positive samples. Frequent domain-motif pairs were filtered out to get specific human-coronavirus pairs. The confidence score for each domain-motif pair template was normalized to a 0-1 scale based on their frequency of occurrence. The domain and motif pairs dataset specific to coronavirus and humans was combined with the DMI template library, and the average confidence scores of the domain and motif pairs were calculated as the final scores (SDMI).    
 
- Obtain the training set of C1C2hC2vC3
  ``` 
  python DMI/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py
  ```
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
  ```   
  python DMI/C1C2C3/10folds_C1223_DMI.py 
  ```

<a name="(S)"></a>
## ⚗️&nbsp; Structure-Based Approaches (S)    
- Similar to the IM method, the S method makes inferences based on the similarity of protein structures. The template dataset used here is the structural data corresponding to the template dataset in the IM method, and is also given the same confidence score (SS) as the IM method. The structural data was obtained from the [PDB](https://www.rcsb.org/) . Structural simulations were performed using [AlphaFold2](https://github.com/google-deepmind/alphafold) for proteins without structural annotations. The relatively advanced [US-align algorithm](https://zhanggroup.org/US-align/) was used to compare the protein structures between virus and virus as well as between virus and human, with TM-score≥0.5 as the judgment criterion for the structural similarity of two proteins.
    
-Obtain the training set of C1C2hC2vC3
  ```
  python S/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py
  ```
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
  ```
  python S/C1C2C3/10folds_C1223_S.py   
  ```


<a name=" Integrated_Model"></a>
## 🧐&nbsp;  Integrated Model
- For a more comprehensive prediction of coronaviruses and human PPIs, we finally constructed an integrated model with RF as a meta learner. This model synthesizes five methods to further improve the performance in the C2v and C3 test sets.
  ```
  python Integrated_model/C1C2C3_RF.py
  python Integrated_model/C1C2C3_LR.py
  python Integrated_model/C1C2C3_NB.py
  python Integrated_model/C1C2C3_SVM.py
  ```
  
<a name="Predicted"></a>
## 📈&nbsp; Predicted
**Threshold_setting**:
```
python final_predict/threshold_setting.py
```   
**Run the Model**:       
```
python final_predict/predict.py
```
<a name="Contact"></a>
## ❤️&nbsp; Contact
For any questions or collaborations, please contact Jia Wang at wang.jia@mail.hzau.edu.cn.
