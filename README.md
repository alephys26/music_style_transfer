# CSL4020: Deep Learning | Course Project
Music Style Transfer via MIDI files, output Beethoven
---
The project uses [Maestro V3.0 dataset](https://magenta.tensorflow.org/datasets/maestro).  
The files included with this project are:
1. **demo.ipynb**  
  The notebook to run all the scripts as and when needed with all the instructions inside.
2. **models.py**  
  Contains the definition of the models.
3. **train.py**  
  Code to train the CycleGAN model. (Currently, the default values are the ones obtained after tuning.)
4. **tune.py**  
  Code to tune the CycleGAN model.
5. **infer.py**  
  Code to get the final accuracies.
6. **train_loader_beethoven.pt, train_loader_chopin.pt, test_loader_beethoven.pt, test_loader_chopin.pt**  
  The various dataloaders used in training and testing the model. 
7. **training_log_last.csv**  
  The log of the best run.
8. **data_preprocessing.ipynb**  
  Run this file only if you want to regenerate the data-loaders.  
9. **report.pdf**
  The report of the Project detailing the architecture and results.


If any issue arises over the pre-trained models, kindly rerun the training script.

> ___Yash Shrivastava, B21CS079___  
  ___Muneshwar Mansi Kailash, B21CS047___  
  ___Chaitanya Gaur, B21ES007___

## References 
1. [Symbolic Music Genre Transfer with CycleGAN](https://arxiv.org/abs/1809.07575)
2. [Audio Input Generates Continuous Frames to Synthesize Facial Video Using Generative Adversarial Networks](https://arxiv.org/abs/2207.08813)
3. [TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up](https://arxiv.org/abs/2102.07074)
4. [Music style migration based on generative Adversarial Networks](https://doi.org/10.1016/j.aej.2024.12.081)  
5. [Generating Music Transition by Using a Transformer-Based Model](https://www.researchgate.net/publication/354665705_Generating_Music_Transition_by_Using_a_Transformer-Based_Model)

