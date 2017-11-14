# SOUND CLASSIFICATION WITH TENSORFLOW ON RASPBERRY PI

![alt text](https://raw.githubusercontent.com/GianlucaPaolocci/Sound-classification-on-Raspberry-Pi-with-Tensorflow/master/img/Immagine.png)


#  BUILD THE PROJECT

  Install following Python libraries on your PC/Workstation and Raspberry Pi:
  
    Tensorflow, Scikit-learn, Librosa
  
  Install following library on your Raspberry only:
  
    Sounddevice

1. **DOWNLOAD UrbanSound8K DATASET**

  https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html

2. **TRAIN THE MODEL**

  Set the right path where you downloaded the dataset in your code.

  Set the right path where you want to save the trained model.

  Run "trainModel.py" on your PC/Workstation.

3. **RUN THE MODEL**

  Export the trained model on you Raspberry Pi ('model.meta', 'model.index', 'checkpoint', 'model.data-00000-of-00001').
  
  Export 'fit_params.npy' on your Raspberry Pi.

  Run "classiPi.py" on your Raspberry and enjoy!

# REMEMBER TO

  Remember to reference this project in your research work.

# AUTHORS
 
  Gianluca Paolocci, University of Naples Parthenope, Science and Techonlogies Departement, Ms.c Applied Computer Science
  https://www.linkedin.com/in/gianluca-paolocci-a19678b6/
  
  Luigi Russo, University of Naples Parthenope, Science and Techonlogies Departement, Ms.c Applied Computer Science
  
# CONTACTS

  if you have problems, questions, ideas or suggestions, please contact me to:
  - **gianluca.paolocci@studenti.uniparthenope.it**
  
 
  

 
