# VAE

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

from omegaconf import OmegaConf

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# plt.figure(figsize=(10, 10))
# for c in range(16):
#     plt.subplot(4, 4, c + 1)
#     plt.imshow(x_train[c].squeeze(), cmap='gray')

# plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0

train_size = int(len(x_train) * 0.9)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=1024)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# print(len(dataset), len(train_dataset), len(val_dataset), len(test_dataset))

train_batch_size = 1024
val_batch_size = 128
test_batch_size = 1024

train_dataloader = train_dataset.batch(train_batch_size, drop_remainder=True)
val_dataloader = val_dataset.batch(val_batch_size, drop_remainder=True)
test_dataloader = test_dataset.batch(test_batch_size, drop_remainder=True)

_cnn_cfg = {
	'name': 'VAE',
	# 'data_normalize': True,
	'latent_dim': 2,
	'enc': {
		'conv1': {
			'filters': 32,
			'kernel_size': 3,
			'strides': [2, 2],
			'activation': 'relu',
		},
		'conv2': {
			'filters': 64,
			'kernel_size': 3,
			'strides': [2, 2],
			'activation': 'relu',
		},
		'out_fc': {
			'units': 4, # latent_dim * 2 (mu, log_var)
		},
	},
	'dec': {
		'in_fc': {
			'units': 7*7*32,
		},
		'reshape_shape': [7, 7, 32],
		'tr_conv1': {
			'filters': 64,
			'kernel_size': 3,
			'strides': [2, 2],
			'padding': 'same',
			'activation': 'relu',
		},
		'tr_conv2': {
			'filters': 32,
			'kernel_size': 3,
			'strides': [2, 2],
			'padding': 'same',
			'activation': 'relu',
		},
		'tr_conv3': {
			'filters': 1,
			'kernel_size': 3,
			'strides': [1, 1],
			'padding': 'same',
		},
	}
}
_cnn_cfg = OmegaConf.create(_cnn_cfg)
# print(OmegaConf.to_yaml(_cnn_cfg))

class VAE(tf.keras.Model):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.latent_dim = cfg.latent_dim

		self.encoder = tf.keras.Sequential([
			tf.keras.layers.Conv2D(**cfg.enc.conv1),
			tf.keras.layers.Conv2D(**cfg.enc.conv2),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(cfg.enc.out_fc.units)
		], name='encoder')

		self.decoder = tf.keras.Sequential([
			tf.keras.layers.Dense(units=cfg.dec.in_fc.units, activation='relu'),
			tf.keras.layers.Reshape(target_shape=tuple(cfg.dec.reshape_shape)),

			tf.keras.layers.Conv2DTranspose(**cfg.dec.tr_conv1),
			tf.keras.layers.Conv2DTranspose(**cfg.dec.tr_conv2),
			tf.keras.layers.Conv2DTranspose(**cfg.dec.tr_conv3)
		], name='decoder')

		self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
		self.recon_loss_tracker = tf.keras.metrics.Mean(name='recon_loss')
		self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')

	@property
	def metrics(self):
		return [
			self.total_loss_tracker,
			self.recon_loss_tracker,
			self.kl_loss_tracker,
		]

	@tf.function
	def sample(self, epsilon=None, sample_size=100):
		if epsilon is None:
			epsilon = tf.random.normal(shape=(sample_size, self.latent_dim))
		return self.decode(epsilon)

	def encode(self, x, training=False):
		mu, logvar = tf.split(
			self.encoder(x, training=training),
			num_or_size_splits=2,
			axis=1
		)
		return mu, logvar

	def reparameterize(self, mu, logvar):
		"""get z"""
		epsilon = tf.random.normal(shape=mu.shape)
		return mu + epsilon * tf.exp(logvar * .5)

	def decode(self, z, training=False):
		return tf.sigmoid(self.decoder(z, training=training))

	def call(self, input, training=False):
		mu, logvar = self.encode(input, training=training)
		z = self.reparameterize(mu, logvar)
		output = self.decode(z, training=training)
		return output, z, mu, logvar

	def train_step(self, data):
		images, _ = data
		# images = [B X 28 X 28] -> [B X 28 X 28 X 1]
		images = tf.cast(tf.expand_dims(images, -1), tf.float32)

		with tf.GradientTape() as tape:
			outputs, z, z_mu, z_logvar = self(images, training=True)

			# reconstuction loss
			recon_loss = tf.reduce_mean(
				tf.reduce_sum(
					tf.keras.losses.mae(images, outputs),
					axis=(1, 2)
				)
			)

			# kld_loss
			kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar))
			kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

			# total_loss
			total_loss = recon_loss + kl_loss

		# compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(total_loss, trainable_vars)

		# update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# update the metrics
		self.total_loss_tracker.update_state(total_loss)
		self.recon_loss_tracker.update_state(recon_loss)
		self.kl_loss_tracker.update_state(kl_loss)

		# tensorboard image update
		tf.summary.image("train_source_img", images, max_outputs=5)
		tf.summary.image("train_recon_img", outputs, max_outputs=5)

		# return a dict mapping metrics names to current values
		logs = {m.name: m.result() for m in self.metrics}

		return logs

	def test_step(self, data):
		images, _ = data
		# images = [B X 28 X 28] -> [B X 28 X 28 X 1]
		images = tf.cast(tf.expand_dims(images, -1), tf.float32)

		outputs, z, z_mu, z_logvar = self(images, training=False)

		# reconstuction loss
		recon_loss = tf.reduce_mean(
			tf.reduce_sum(
				tf.keras.losses.mae(images, outputs),
				axis=(1, 2)
			)
		)

		# kld_loss
		kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar))
		kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

		# total_loss
		total_loss = recon_loss + kl_loss

		# update the metrics
		self.total_loss_tracker.update_state(total_loss)
		self.recon_loss_tracker.update_state(recon_loss)
		self.kl_loss_tracker.update_state(kl_loss)

		# tensorboard image update
		tf.summary.image("val_source_img", images, max_outputs=5)
		tf.summary.image("val_recon_img", outputs, max_outputs=5)

		# return a dict mapping metrics names to current values
		logs = {m.name: m.result() for m in self.metrics}

		return logs

max_epoch = 50
learning_rate = 1e-3

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

model = VAE(_cnn_cfg)

model.build((1, 28, 28, 1))
model.summary()

model.compile(
	optimizer=optimizer
)

tf.config.run_functions_eagerly(True) # DEBUG

model.fit(
	train_dataloader,
	validation_data=val_dataloader,
	epochs=max_epoch
)

def get_latent_img(model, n, single_img_size=28):
	norm = tfp.distributions.Normal(0, 1)
	grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
	grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
	width = single_img_size * n
	height = width
	image = np.zeros((height, width))

	for i, yi in enumerate(grid_y):
		for j, xi in enumerate(grid_x):
			z = np.array([[xi, yi]])
			x_decoded = model.sample(z)
			digit = tf.reshape(x_decoded[0], (single_img_size, single_img_size))
			image[i*single_img_size:(i + 1)*single_img_size,
				  j*single_img_size:(j + 1)*single_img_size] = digit.numpy()
	return image

latent_img = get_latent_img(model, n=20)
plt.figure(figsize=(10, 10))
plt.imshow(latent_img, cmap='Greys_r')
plt.axis('Off')
plt.show()
