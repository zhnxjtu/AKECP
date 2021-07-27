# AKECP

This is the code for "AKECP: Adaptive Knowledge Extraction from Feature Maps for Fast and Efficient Channel Pruning". We take the code for ResNet-110 as an axample.

Firstly, in floder "python code", run "Main_FmObtain.py" to get the feature maps (Here, we input 10 images as an example, where the first image is used for pruning in the paper); 

Secondly, in floder "matlab code", run "run.m" to get the pruning indices;

Finally, in floder "python code", run "res110_pruning_trainer.py" to restore the accuracy.


The pruned model with four different conpression ratios (40%, 60%, 70%, 80%) are accessed in  'code\python code\Pruned model' 
