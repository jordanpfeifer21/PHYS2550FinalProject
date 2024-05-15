## Anomaly Detection in Particle Physics Using AI at the LHC


*Jordan Pfeifer, Ekin Secilmis, Egor Serebriakov*

### About the data used in the project

We have three "black boxes'' meant to be representative of the actual data from LHC. Each "black box" contains 1M events. The given events might have signals that we consider as anomaly signals.

Additionally, we have a background sample of 1M events simulated using Pythia8 and Delphes. This data was simulated in order to aid in the anomaly detection from the "black boxes". However, some assumption during the simulation might not exactly reflect the "black boxes" data.

All datasets are stored as pandas DataFrames saved to compressed h5 format. Each event consists of 700 particles (we might have some events with some degree of zero padding) and each particle has three coordinates (pT, eta, phi).

> [*Original dataset*](https://paperswithcode.com/dataset/lhc-olympics-2020)

### Links

1. [Google Slides](https://docs.google.com/presentation/d/1tQCU03cHe44oVzg05qOPGl-vTHJSb7w3zzfLtAVs7Js/edit?usp=sharing)

1. [Overleaf Report](https://www.overleaf.com/7219896992pncptqkkxrfz#d5daf8)

1. [Google Drive with Split Data ($\texttt{TensorFlow}$) and Preprocessed Data ($\texttt{torch})](https://drive.google.com/drive/folders/1MGpipM4VPwxlxCE4IA5QRMixz0L2cg88?usp=sharing)

   > Files *X_train_small.csv, X_test_small.csv, X_valid_small.csv* are smaller versions of background data that can be useful to build an appropriate model faster. *X_test_first.csv* is the data from the first box and so on.

1. [Google Drive with Model Parameters and Final Graphs](https://drive.google.com/drive/folders/1yE34yUAKLIokuPGRGah8HkCiBszSFL0y?usp=sharing)
