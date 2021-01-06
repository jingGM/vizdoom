from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.sac.policies import FeedForwardPolicy as SACForward
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
import tensorflow as tf
import warnings
from itertools import zip_longest
import numpy as np


def customizedCNN(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=8, filter_size=6, stride=3, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=8, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=8, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_4 = conv_to_fc(layer_3)
    layer_5 = activ(linear(layer_4, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    layer_6 = activ(linear(layer_5, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))

    active = tf.tanh
    pi = active(linear(layer_6, "pi_fc{}".format(1), 64, init_scale=np.sqrt(2)))
    # pi = active(linear(pi, "pi_fc{}".format(2), 128, init_scale=np.sqrt(2)))
    vf = active(linear(layer_6, "vf_fc{}".format(1), 64, init_scale=np.sqrt(2)))
    # vf = active(linear(vf, "vf_fc{}".format(2), 128, init_scale=np.sqrt(2)))
    return pi, vf


def SACCNN(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=8, filter_size=6, stride=3, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=8, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=8, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_4 = conv_to_fc(layer_3)
    layer_5 = activ(linear(layer_4, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    layer_6 = activ(linear(layer_5, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))

    active = tf.tanh
    pi = active(linear(layer_6, "pi_fc{}".format(1), 64, init_scale=np.sqrt(2)))
    # pi = active(linear(pi, "pi_fc{}".format(2), 128, init_scale=np.sqrt(2)))
    return pi

def simpleCNN(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=3, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=5, stride=3, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=128, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_4 = conv_to_fc(layer_3)
    layer_5 = activ(linear(layer_4, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    layer_6 = activ(linear(layer_5, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))

    active = tf.tanh
    pi = active(linear(layer_6, "pi_fc{}".format(1), 64, init_scale=np.sqrt(2)))
    # pi = active(linear(pi, "pi_fc{}".format(2), 128, init_scale=np.sqrt(2)))
    return pi


class SACPolicy(SACForward):
    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(SACPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", cnn_extractor=simpleCNN, **_kwargs)


class PPOPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, feature_extraction="cnn",
                 **kwargs):
        super(PPOPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                        scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent, vf_latent = customizedCNN(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs))

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


def mlp_extractor(flat_observations):
    net_arch = [1024, dict(vf=[512, 256, 256], pi=[512, 256, 256]), 128, 64]
    act_fun = tf.tanh

    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value
