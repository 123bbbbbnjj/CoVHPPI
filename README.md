# Prediction and Evaluation of Coronavirus and Human Protein-Protein Interactions

## Overview
The high lethality and infectiousness of coronaviruses, particularly SARS-Cov-2, pose a significant threat to human society. Understanding coronaviruses is crucial for mitigating the coronavirus pandemic. In this study, we conducted a comprehensive comparison and evaluation of five prevalent computational methods: interolog mapping, domain-domain interaction methodology, domain-motif interaction methodology, structure-based approaches, and machine learning techniques. These methods were assessed using unbiased datasets that include C1, C2h, C2v, and C3 test sets. Ultimately, we integrated these five methodologies into a unified model for predicting protein-protein interactions (PPIs) between coronaviruses and human proteins. Based on this model, we further established a high-confidence PPI network between coronaviruses and humans, consisting of 18,012 interactions between 3,843 human proteins and 129 coronavirus proteins. The reliability of our predictions was further validated through current knowledge framework and network analysis.


## Software Environment
- **Python**: 3.12.2
- **Keras**: 3.1.1
- **NumPy**: 1.26.4
- **Pandas**: 2.2.1
- **scikit-learn**: 1.4.1
- **TensorFlow**: 2.16.1
- **XGBoost**: 2.0.3
- **zzd**: 1.0.5
## Model download
1.EsmMean（2560）:https://ai.gitee.com/hf-models/facebook/esm2_t36_3B_UR50D    
2.ProtTrans（1024）:https://github.com/agemagician/ProtTrans    
3.doc2vec（32）:http://zzdlab.com/intersppi/hvppi/download/HVPPI.tar.gz    
  
## Datasets
- **C1**: C1 represents the regular randomized partition of the test set; 
- **C2v and C2h**: C2v and C2h due to the involvement of two different species, virus and human
- **C3**: C3 represents the fact that none of the proteins in the test set can be found in the training set. 
- **Fold5**:five-fold cross-validation comparisons

## Five Computational-Based Methods for PPI Prediction

### Machine Learning Techniques (ML)
- **Feature Extraction**:
  - Amino acid composition (AAC) and order: `python ML/features/CKSAAP.py`
  - Evolutionary information: `python ML/features/get_pssm.py`
  - Protein embeddings: 
    - `python ML/features/doc2vec.py`
    - `python ML/features/EMS2.py`
    - `python ML/features/prottrans.py`
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:
  - `python 10folds_RF_C1223.py cksaap ctdc ctdt ctdd rpssm EsmMean`

### Interolog Mapping (IM)
- The IM method is mainly based on the homology of protein sequences for inference 
- Obtain the training set of C1C2hC2vC3    
- `python IM/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py`
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:    
- `python IM/C1C2C3/10folds_C1223_IM.py`

### Domain-Domain Interaction Methodology (DDI)
- The DDI method predicts the interaction probability of query protein pairs based on the detection of interacting domain-domain pairs
-Obtain the training set of C1C2hC2vC3    
-  `python DDI/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py`
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:    
- `python DDI/C1C2C3/10folds_C1223_DDI.py`

### Domain-Motif Interaction Methodology (DMI)
- The DMI method, which is based on identifying interacting human protein domain and viral protein motif pairs for interaction probability prediction, is considered the most biologically meaningful method
-Obtain the training set of C1C2hC2vC3    
-  `python DMI/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py`
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:    
- `python DMI/C1C2C3/10folds_C1223_DMI.py`   

### Structure-Based Approaches (S)   
- Compared to sequences, there is a lack of data for structures, although the relative conservatism of structures favors the prediction of PPIs. Similar to the IM method, the S method makes inferences based on the similarity of protein structures    
-Obtain the training set of C1C2hC2vC3   
- `python S/C1C2C3/VHPPI_C1C2C3/get_vhppi_c1c2c3.py`    
- **Model Testing on C1, C2h, C2v, and C3 Test Sets**:    
- `python S/C1C2C3/10folds_C1223_S.py`   

## Integrated Model   
In order to comprehensively predict the interaction between coronaviruses and human proteins, we ultimately constructed an ensemble model using RF as a meta learner by integrating the output probabilities of five methods    
-`python Integrated_model/C1C2C3_RF.py`   
## Predicted   
**Run the Model**:    
-`python final_predict/predict.py`

## Contact
For any questions or collaborations, please contact Jia Wang at wang.jia@mail.hzau.edu.cn.
