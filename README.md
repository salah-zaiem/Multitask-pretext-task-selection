# ICLR2022 submission
Here is the repository to reproduce the results of our iclr22 submission. It involves four major steps : pseudo-labels extraction, pretext tasks selection and weighting, self-supervised training, downstream finetuning. 


### Feature extraction
The extraction is done with the Opensmile library, all the files concerned are in the folder "data\_preparation/" and it involves three steps : 
- Extracting the signal features using the two scripts 
- Mixing the two sets of features
- Normalizing the values per feature

You can run the three steps after downloading the CommonVoice files from the dedicated Mozilla website through running data_preparation/preparation.sh
### Pretext task selection and weighting 

Inside the exp_optimizing.py file, you can change the number of epochs for optimization and the function used (softmax/sparsemax) according to either you want a sparse or not weighting for your considered pretext tasks.

If you want to look for the best weighting using sparsemax on librispeech you would have to change the variable sparse in exp_optimizing.py to False then specify the number of iterations  wanted and run :
'''
python exp_optimizing.py Libri_K_matrices/ Libri_logL_matrices/ 0 Lresults/
'''
The result of this selection will be output in Lresults in a file named after the date and time of the experiment. You can change the 0 parameters by any value to get the verbose version printing the values of the weights after every iteration.  

### Self-supervised training

Pretrained models for the CommonVoice pretraining  are available here : 
https://1drv.ms/u/s!AtZNOLRhbqF6aH0KE5qIbwzEf60?e=NTaFDL

To launch a self-supervised learning, you will need the audio files form CommonVoice, the extracted workers from the first part, 

### Downstream finetuning

##### LibriSpeech 


For Librispeech, we perform end-to-end retraining and testing in one step. You'll have to copy the folder resulting from the pretraining in the Librispeech retraining folder first.

Then example.sh provides an example for retraining. At the end of the retraining, the test WER is output. 


##### VoxCeleb

Speaker Recognition is a two-step action in our work. First, we train Xvectors, as stated in the training\_xvectors\_example.sh. Afterwards, a few changes have to be made to the retraining yamls, mainly links to the embedding model. An example is provided with AlphaRatio. An example for verification and final results computing is provided in verification.sh  


CommonVoice pretrained models can be used for the downstream tasks, by copying the folder in the Downstream task folder

VoxCeleb pretrained models can be used for the verification phase, by running a command similar to the one given in verification.sh 

In both cases, you can run the retraining by following the examples given in the .sh files. 

