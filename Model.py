
class Vae_cnn_BILSTM:

    def __init__(self, n_features, latent_dim, n_runs,
                 n_level_conv, filters, window_conv, beta):
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_runs = n_runs
        self.n_level_conv = n_level_conv
        self.filters = filters
        self.window_conv = window_conv
        self.beta = beta
        
        
    def set_train_matrix_boat(self):
        # Initializing train Matrix
        self.train_matrix, self.labels = self.prepare_training(self.path, n_runs)
        self.labels = np.array(self.labels)
        noise = np.random.normal(loc=0, scale=0.5, size=self.train_matrix.shape)
        self.train_matrix_noisy = self.train_matrix + noise
        self.masks_train = [self.return_mask(num, np.array(self.labels)) for num in range(0, 9)]

        
    def set_train_matrix_airplane(self,train_matrix_airplane, labels):
        self.labels = np.array(labels)
        self.train_matrix = train_matrix_airplane
        noise = np.random.normal(loc=0, scale=2, size=self.train_matrix.shape)
        self.train_matrix_noisy = self.train_matrix + noise
        self.masks_train = [self.return_mask(num, np.array(labels)) for num in range(0, np.max(labels)+1)]        
  

    def set_train_matrix_strands(self, train_matrix_strands, labels):
        self.labels = np.array(labels)
        self.train_matrix = train_matrix_strands
        noise = np.random.normal(loc=0, scale=2, size=self.train_matrix.shape)
        self.train_matrix_noisy = self.train_matrix + noise
        self.masks_train = [self.return_mask(num, np.array(labels)) for num in range(0, np.max(labels)+1)]        
          

    # Take csvs and return training matrix
    def prepare_training(self, path, n_runs):
        labels = []

        def closest_4(n, m):
            q = n / m
            n1 = m * q
            if (n * m) > 0:
                n2 = m * (q + 1)
            else:
                n2 = m * (q - 1)
            if abs(n - n1) < abs(n - n2):
                return int(n1)
            return int(n2)

        def extend_line(run, max_len):
            difference = abs(len(run) - max_len)
            extension = np.array([run[-1]] * difference)

            if difference != 0:
                run = np.vstack([run, extension])
            return run

        def get_max_len(sequence_list):
            max_len = 0
            min_len = 1000
            for seq in sequence_list:
                if len(seq) > max_len:
                    max_len = len(seq)
                if len(seq) < min_len:
                    min_len = len(seq)
            return max_len, min_len

        def construct_matrix(sequence_list):
            max_len, min_len = get_max_len(sequence_list)
            len = closest_4(max_len, 4)
            len = 440
            train_matrix = np.zeros(shape=(n_runs, len, self.n_features))
            for index, run in enumerate(sequence_list):
                line = extend_line(run, len)
                train_matrix[index] = line
            return train_matrix

        def stadard_sequences(seqs):
            for i, seq in enumerate(seqs):
                # seqs[i] = MinMaxScaler(feature_range=[0,1]).fit_transform(seq)
                seqs[i] = StandardScaler().fit_transform(seq)
            return seqs

        def order_runs_by_len(runs):
            runs.sort(key=len)
            for r in runs:
                labels.append(r['Choice'][0])
            for i, r in enumerate(runs):
                runs[i] = runs[i].drop(columns=['Choice'])
            return runs, labels

        def read_sequences():
            run_list_mix = []
            for index in range(n_runs):
                run_csv = pd.read_csv(path + str(index))
                run_csv = run_csv.drop(columns=['Unnamed: 0'])
                run_list_mix.append(run_csv)
            run_list_ordered, labels = order_runs_by_len(run_list_mix)
            stands = stadard_sequences(run_list_ordered)
            padded_matrix = construct_matrix(stands)
            return padded_matrix, labels

        return read_sequences()

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
            if i == 0:
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

        for i in range(1):
            x_dec = Conv1DTranspose(input_tensor=x_dec, filters=self.filters,
                                    kernel_size=self.window_conv, dil_rate=1, padding='same')
            x_dec = UpSampling1D(size=2)(x_dec)

        decoder_dense = Dense(self.n_features, name="Decoder_output")
        decoder_outputs = decoder_dense(x_dec)

        self.decoder = Model(decoder_inputs, decoder_outputs)
        if summary:
            self.decoder.summary()

        decoder_outputs = self.decoder(self.encoder.output[2])
        decoder_outputs = self.decoder(self.encoder(encoder_inputs)[2])

        # The number of epochs at which KL loss should be included
        klstart = 200
        # number of epochs over which KL scaling is increased from 0 to 1
        kl_annealtime = 50

        class AnnealingCallback(Callback):
            def __init__(self, weight):
                self.weight = weight

            def on_epoch_end(self, epoch, logs={}):
                if epoch > klstart:
                    new_weight = min(K.get_value(self.weight) + (1. / kl_annealtime), 1)
                    K.set_value(self.weight, new_weight)
                print("Current KL Weight is " + str(K.get_value(self.weight)))

        # Starting value
        weight = K.variable(0.)

        def vae_loss(weight):
            def loss(y_true, y_pred):
                # mse loss
                reconstruction_loss = mse(K.flatten(y_true), K.flatten(y_pred))
                reconstruction_loss *= self.n_features * self.train_matrix.shape[1]
                kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
                kl_loss = K.mean(kl_loss, axis=-1)
                kl_loss *= -0.5
                return K.mean(reconstruction_loss + self.beta * kl_loss)

            return loss

        self.model = Model(encoder_inputs, decoder_outputs)
        self.model.compile(optimizer='adam', loss=vae_loss(weight))

        if summary:
            self.model.summary()
        #return (self.model, self.encoder, self.decoder)

    def train_model(self, n_epochs, verbose):
        self.model.fit(self.train_matrix_noisy, self.train_matrix,
                  epochs=n_epochs,
                  batch_size=200,
                  verbose=verbose,
                  validation_split=0.1)

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
            markers = ['o','o','o','o','o','o','o','o','o','X','X','X','X','X','X','X','X','X','X','X']
            for index, mask in enumerate(self.masks_train):
                ax.scatter(principalComponents[:, 0][mask],
                           principalComponents[:, 1][mask],
                           principalComponents[:, 2][mask], marker=markers[index])

            plt.legend(labels=np.arange(0, np.max(self.labels)))
            plt.title(str(title))
            plt.show()
            for index,mask in enumerate(self.masks_train):
                plt.scatter(x=principalComponents[:, 0][mask],
                            y=principalComponents[:, 1][mask], marker=markers[index])

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
    
