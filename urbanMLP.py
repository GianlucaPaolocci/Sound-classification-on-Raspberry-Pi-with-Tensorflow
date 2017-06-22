
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from tempfile import TemporaryFile
import time

'''0 = air_conditioner
1 = car_horn
2 = children_playing
3 = dog_bark
4 = drilling
5 = engine_idling
6 = gun_shot
7 = jackhammer
8 = siren
9 = street_music
10 = truck'''


def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()


sound_file_paths = ["30204.wav", "2937.wav"]
sound_names = ["air_conditioner", "car_horn"]

#raw_sounds = load_sound_files(sound_file_paths)

#plot_waves(sound_names,raw_sounds)
#plot_specgram(sound_names,raw_sounds)
#plot_log_power_specgram(sound_names,raw_sounds)


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    #print file_name
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz



def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    try:
        n = np.load('/home/shared/SM_Paolocci_Russo/SM/names.npy')
        l = np.load('/home/shared/SM_Paolocci_Russo/SM/labels.npy')
        f = np.load('/home/shared/SM_Paolocci_Russo/SM/features.npy')
        print("Files found!")
    except:
        f, l, n = np.empty((0, 193)), np.empty(0), np.empty(0)

    features, labels, name = np.empty((0,193)), np.empty(0), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        print sub_dir

        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            index = np.where(n == fn)
            i = index[0]
            if i.shape[0] > 0:
                #print("in if")
                ii = i[0]
                features = np.vstack([features, f[ii]])
                labels = np.append(labels, l[ii])
                name = np.append(name, fn)
            else:
                #print("in else")
                name = np.append(name, fn)
                mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])

                if "truck" in fn:
                    labels = np.append(labels, "10")
                    #labels = np.append(labels, fn.split('-')[1])
                    #print "truck found"
                else:
                    labels = np.append(labels, fn.split('-')[1])
                    #print fn.split('-')[1]

                print ("Sono: ",sub_dir," e adesso scrivo nei file")
                with open('/home/shared/SM_Paolocci_Russo/SM/names.npy', 'a') as f1:
                    np.save(f1,name)
                with open('/home/shared/SM_Paolocci_Russo/SM/labels.npy', 'a') as f2:
                    np.save(f2, labels)
                with open('/home/shared/SM_Paolocci_Russo/SM/features.npy', 'a') as f3:
                    np.save(f3, features)
    #  print("Shape Features: "),np.array(features).shape
    return np.array(features), np.array(labels, dtype = np.int), np.array(name)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


parent_dir = '/home/shared/SM_Paolocci_Russo/SM/UrbanSound8K/audio'

sub_dirs = ['fold1', 'fold2', 'fold11', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']
test_dir = ['test_sound/truck']

'''
ff1=open('/home/shared/SM_Paolocci_Russo/SM/features.npy','rw+')
ff2=open('/home/shared/SM_Paolocci_Russo/SM/names.npy','rw+')
ff3=open('/home/shared/SM_Paolocci_Russo/SM/labels.npy','rw+')


ff1.truncate()
ff2.truncate()
ff3.truncate()

ff1.close()
ff2.close()
ff3.close()
'''

features, labels, name = parse_audio_files(parent_dir,sub_dirs)
#features_t, labels_t, name_t = parse_audio_files(parent_dir,test_dir)

f = np.load('/home/shared/SM_Paolocci_Russo/SM/features.npy')
n = np.load('/home/shared/SM_Paolocci_Russo/SM/names.npy')
l = np.load('/home/shared/SM_Paolocci_Russo/SM/labels.npy')

#print(n)
#print(l)
#print(f)

time.sleep(2)


labels = one_hot_encode(labels)
#labels_t = one_hot_encode(labels_t)

train_test_split = np.random.rand(len(features)) < 0.70
train_x = features[train_test_split]
train_y = labels[train_test_split]
test_x = features[~train_test_split]
test_y = labels[~train_test_split]


# --------------------------------------------------------------------------------------------------------------------------

# #### Training Neural Network with TensorFlow


import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support



training_epochs = 50000
n_dim = features.shape[1]
n_classes = 11
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01
model_path = "/home/shared/SM_Paolocci_Russo/SM/model"

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2 )


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
saver = tf.train.Saver()#[X, Y, W_1, b_1, h_1, W_2, b_2, h_2, W, b, y_]

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:train_y})
        cost_history = np.append(cost_history,cost)
        print "Epoch: ", epoch, " cost ", cost

    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_x})
    y_true = sess.run(tf.argmax(test_y,1))

    #y_pred_t = sess.run(tf.argmax(y_, 1), feed_dict={X: features_t})
    #print ("truck:") ,y_pred_t

    #saving model
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

'''fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()'''

p,r,f,s = precision_recall_fscore_support(y_true, y_pred)#, average='micro')
print ("F-Score:"), f
print ("Precision:"), p
print ("Recall:"), r
print ("Support:"), s






