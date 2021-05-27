# %% [code]
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv1D
from keras.models import Model
from keras import backend as K
from sklearn.utils import resample
import keras
from keras import objectives
from keras.objectives import mse
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv1D, Conv2DTranspose, Lambda, MaxPooling1D, UpSampling1D, RepeatVector, \
    Dropout, BatchNormalization, Bidirectional, GlobalMaxPooling1D
import keras.backend as K
from keras.optimizers import Adam
from tensorflow_core.python.training import optimizer
from keras.callbacks import Callback
from keras import optimizers
from keras.regularizers import l2
from keras.layers import GlobalAveragePooling1D
from datetime import datetime
from keras.layers import concatenate, average, minimum, maximum, add, dot, AveragePooling1D, MaxPooling1D
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Vae_cnn_BILSTM:
    def __init__(self, n_features, latent_dim,n_level_conv, filters, window_conv, beta):#, kl_start, kl_anneal_time, kl_anneal):
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_level_conv = n_level_conv
        self.filters = filters
        self.window_conv = window_conv
        self.beta = beta
#         self.kl_start = kl_start
#         self.kl_anneal_time = kl_anneal_time
#         self.kl_anneal = kl_anneal
        self.weight = K.variable(0.)

    def set_train_matrix(self, train_matrix, labels):
        self.labels = np.array(labels)
        self.train_matrix = train_matrix
        self.n_runs = len(train_matrix)  
        mat_non_zero = self.train_matrix != 0
        noise = np.random.normal(0, 1, self.train_matrix.shape)
        self.train_matrix_noisy = np.zeros(shape=self.train_matrix.shape)
        self.train_matrix_noisy[mat_non_zero] = self.train_matrix[mat_non_zero] + noise[mat_non_zero]
        print(self.train_matrix_noisy.shape)
        self.masks_train = [self.return_mask(num, np.array(labels)) for num in range(0, np.max(labels)+1)]        
          

    def create_model(self, summary):
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean=0 and std=1.0
            epsilon = K.random_normal(shape=(batch, dim), mean=K.mean(z_mean), stddev=1.)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        
        
        def Conv1DTranspose(input_tensor, filters, kernel_size, dil_rate, strides=1, padding='same'):
            x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
            x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1),
                                strides=(strides, 1), padding=padding,
                                activation='relu',
                                dilation_rate=dil_rate)(x)
            x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
            return x
        
       
        
        encoder_inputs = Input(shape=(self.train_matrix.shape[1], self.n_features), name='Encoder_input')        
        x = encoder_inputs

        for i in range(self.n_level_conv):
            x = Conv1D(filters=self.filters,
                       kernel_size=self.window_conv,
                       activation='relu',
                       dilation_rate=1,
                       padding='same')(x)
            x = MaxPooling1D(pool_size=2)(x)
            #x = GlobalMaxPooling1D()(x)
            #if i == 0:
            shape = K.int_shape(x)

        encoder_outputs = Bidirectional(LSTM(self.latent_dim))(x)

        z_mean = Dense(self.latent_dim, name='z_mean', )(encoder_outputs)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(encoder_outputs)
        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        if summary:
            self.encoder.summary()

        decoder_inputs = Input(shape=(self.latent_dim,), name='latent_inputs')
        latent_inputs = Dense(shape[1] * shape[2])(decoder_inputs)
        latent_inputs = Reshape((shape[1], shape[2]))(latent_inputs)
        x_dec = latent_inputs
        decoder_lstm = Bidirectional(LSTM(self.latent_dim, go_backwards=True, return_sequences=True))
        x_dec = decoder_lstm(latent_inputs)

        for i in range(self.n_level_conv):
            x_dec = UpSampling1D(size=2)(x_dec)
#             x_dec = Conv1DTranspose(input_tensor=x_dec, filters=self.filters,
#                                     kernel_size=self.window_conv, dil_rate=1, padding='same')
            x = Conv1D(filters=self.filters,
                       kernel_size=self.window_conv,
                       activation='relu',
                       dilation_rate=1,
                       padding='same')(x_dec)

        # RECONSTRUCTION PROB
#         dec_mean = Dense(self.n_features, name='Dec_mean')(x_dec)
#         dec_log_var = Dense(self.n_features, name='Dec_var')(x_dec)
#         print(K.int_shape(dec_mean), K.int_shape(dec_log_var))
#         decoder_dense_sampled = Lambda(sampling_dec, output_shape=(self.train_matrix.shape[1],self.n_features,), name='reconstr_sampled')([dec_mean, dec_log_var])
        
#         decoder_outputs = decoder_dense_sampled
        #END RECONSTRUCTION PROB
            
        decoder_dense = Dense(self.n_features, name="Decoder_output")
        decoder_outputs = decoder_dense(x_dec)

        self.decoder = Model(decoder_inputs, decoder_outputs)
        if summary:
            self.decoder.summary()

        decoder_outputs = self.decoder(self.encoder.output[2])
        decoder_outputs = self.decoder(self.encoder(encoder_inputs)[2])

        # The number of epochs at which KL loss should be included
        # number of epochs over which KL scaling is increased from 0 to 1

        class AnnealingCallback(Callback):
            def __init__(self, weight):
                self.weight = weight
                
            def on_epoch_end(self, epoch, logs={}):
                if epoch > self.klstart:
                    new_weight = min(K.get_value(self.weight) + (1. / self.kl_annealtime), 1)
                    K.set_value(self.weight, new_weight)
                print("Current KL Weight is " + str(K.get_value(self.weight)))

        # Starting value
        
        def vae_loss(weight):
            def loss(y_true, y_pred):
                # mse loss
                reconstruction_loss = mse(K.flatten(y_true), K.flatten(y_pred))
                reconstruction_loss *= self.n_features * self.train_matrix.shape[1]
                kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
                kl_loss = K.mean(kl_loss, axis=-1)
                kl_loss *= -0.5                
#                 if self.kl_anneal:
#                     return K.mean(reconstruction_loss + self.weight * kl_loss)
#                 else:
                return K.mean(reconstruction_loss + self.beta * kl_loss)
            return loss

        self.model = Model(encoder_inputs, decoder_outputs)
        self.model.compile(optimizer='adam', loss=vae_loss(self.weight))

        if summary:
            self.model.summary()
        #return (self.model, self.encoder, self.decoder)

    def train_model(self, n_epochs, verbose):
        self.model.fit(self.train_matrix_noisy, self.train_matrix,
                  epochs=n_epochs,
                  batch_size=200,
                  verbose=verbose,
                  validation_split=0.1)#,
                  #callbacks=[AnnealingCallback(self.weight)])

    # model.save_weights("Models/Weights/CONV1d_LSTM_STATES_400_runs_08_param.hdf5")
    # model.load_weights("Models/Weights/Model_6_feat_400_lotep2.hdf5")

    def return_mask(self, num, labs):
        arg = np.squeeze(np.argwhere(labs == num))
        return arg

    def plot_encodings(self, return_mean):
        encodings = self.encoder.predict(self.train_matrix)
        enc_mean, enc_var, z_enc = encodings[0], encodings[1], encodings[2]
        print(enc_mean.shape, enc_var.shape, z_enc.shape)

        def plot_pca(title, i):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            markers = ['o','o','o','o','o','o','o','o','o','X','X','X','X','X','X','X','X','X','X','X','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o','o',]
            for index, mask in enumerate(self.masks_train):
                ax.scatter(principalComponents[:, 0][mask],
                           principalComponents[:, 1][mask],
                           principalComponents[:, 2][mask], marker=markers[index])

            plt.legend(labels=np.arange(0, np.max(self.labels)))
            plt.title(str(title))
            plt.show()
            for index,mask in enumerate(self.masks_train):
                plt.scatter(x=principalComponents[:, 0][mask],
                            y=principalComponents[:, 1][mask])

                # break

            plt.legend(labels=np.arange(0, np.max(self.labels)))
            plt.title(str(title))
            plt.show()

        enc_list = [enc_mean, enc_var, z_enc]
        titles = ["MEAN", "LOG_VAR", "SAMPLED"]

        for i, enc in enumerate(enc_list):
            scaler = StandardScaler()
            enc_input = scaler.fit_transform(enc)
            pca = PCA(3)
            principalComponents = pca.fit_transform(enc_input)
            print("Component 1 shape", principalComponents.shape)
            print("PCA variance ratio", pca.explained_variance_ratio_)
            plot_pca('Sequences' + titles[i], 0)

        if return_mean:
            pca = PCA(3)
            principalComponents = pca.fit_transform(enc_mean)
        else:
            pca = PCA(3)
            principalComponents = pca.fit_transform(z_enc)

        return principalComponents
    
    def encode(self,values):
        return self.encoder.predict(values)
    
    def decode(self, encoding):
        return self.decoder.predict(values)
    
    def autoencode(values):
        return self.model.predict(values)
    
    
    def plot_reconstruction_error(self):
        
        #CALCULATE RECONSTRUCTION ERROR AS AVERAGE FOR ALL POINTS (ON TRAIN AND ON UNSEEN)
        reconstruction = self.model.predict([self.train_matrix])
        
        def mse_train(n_train):
            errors_train = []
            for i in range(n_train):
                errors_train.append(abs(self.train_matrix[i] - reconstruction[i]))     
            return np.array(errors_train)

        train_error = mse_train(self.n_runs)
        train_error_avg = np.mean(train_error, axis=2)
        self.train_error_avg = np.mean(train_error_avg, axis=1)

        
        
        for mask in self.masks_train:
            plt.scatter(np.linspace(1,self.n_runs,self.n_runs)[mask],self.train_error_avg[mask])
        plt.title('ERROR ON TRAIN')
        plt.show()
    
