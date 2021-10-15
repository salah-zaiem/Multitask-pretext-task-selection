## Selection and weighting of tasks
##### Computing the K and L matrices

You should follow the examples provided in the CI_estimator/ folder to compute the Ks and Ls matrices corresponding, respectively, to the similarity matrices between speech samples and between their pseudo-labels. A code is available for every dataset involved in the paper. 

##### Computing the weights

Inside the exp_optimizing.py file, you can change the number of epochs for optimization and the function used (softmax/sparsemax) according to either you want a sparse or not weighting for your considered pretext tasks.

If you want to look for the best weighting using sparsemax on librispeech you would have to change the variable sparse in exp_optimizing.py to False then specify the number of iterations  wanted and run :
'''
python exp_optimizing.py Libri_K_matrices/ Libri_logL_matrices/ 0 Lresults/
'''
The result of this selection will be output in Lresults in a file named after the date and time of the experiment. You can change the 0 parameters by any value to get the verbose version printing the values of the weights after every iteration.  


