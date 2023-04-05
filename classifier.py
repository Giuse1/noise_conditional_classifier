import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
from clu import metrics
from flax import struct  # Flax dataclasses
import optax  # Common loss functions and optimizers
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import tensorflow_datasets as tfds
import datasets
from models import wideresnet_noise_conditional
from sampling import *
from flax.training import train_state  # Useful dataclass to keep train state
from sde_lib import *
from configs.vp import cifar10_ddpmpp_continuous as configs
import collections
import pickle
import jax

root_path = "/home/gdigiacomo/score_sde/workdir_cifar_correct"

handler = logging.FileHandler(f"{root_path}/checkpoints_classifier_1e-3/cnn_log.txt",mode='w')
logger = logging.getLogger("log")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

num_epochs = 100
batch_size = 128
shuffle_buffer_size = 10000
prefetch_size = tf.data.experimental.AUTOTUNE
per_device_batch_size = batch_size // jax.device_count()
batch_dims = [per_device_batch_size]

learning_rate = 0.01
momentum = 0.9
config = configs.get_config()
random_seed = 0
tf.random.set_seed(random_seed)
rng = jax.random.PRNGKey(random_seed)
rng, score_rng = jax.random.split(rng)
rng, score_state_rng = jax.random.split(rng)
rng, cnn_state_rng = jax.random.split(rng)


cnn = wideresnet_noise_conditional.WideResnet(
    blocks_per_group=4,
    channel_multiplier=10,
    num_outputs=10)

"""# Save checkpoint function"""
def save_checkpoint_dict(d, epoch):
    def map_nested_dicts(ob, func):
        if isinstance(ob, collections.Mapping):
            return {k: map_nested_dicts(v, func) for k, v in ob.items()}
        else:
            return func(ob)

    ud = flax.core.frozen_dict.unfreeze(d)
    new_ud = map_nested_dicts(ud, lambda v: np.array(v))
    with open(f"{root_path}/checkpoints_classifier_1e-3/classifier_{epoch}",
              'wb') as pickle_file:
        pickle.dump(new_ud, pickle_file)


"""# Diffusion model"""
# score_ckpt_filename = f"{root_path}/checkpoints/checkpoint_26"
sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)

# score_model, init_model_state, initial_params = mutils.init_model(score_rng, config)
# optimizer = losses_lib.get_optimizer(config).create(initial_params)
#
# state_diff = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
#                           model_state=init_model_state,
#                           ema_rate=config.model.ema_rate,
#                           params_ema=initial_params,
#                           rng=score_state_rng)  # pytype: disable=wrong-keyword-args
# sigmas = mutils.get_sigmas(config)
print(f"config.data.centered: {config.data.centered:} ")
scaler = datasets.get_data_scaler(config)
# inverse_scaler = datasets.get_data_inverse_scaler(config)
# score_state = utils.load_training_state(score_ckpt_filename, state_diff)
sampling_eps = 1e-3


"""## 2. Loading data"""
dataset_builder = tfds.builder('cifar10')
split_diffusion, _ = tfds.even_splits('train', n=2)  # _ is for split_FL
del _
train_split_name = split_diffusion
eval_split_name = 'test'


def preprocess_fn(d):
    """Basic preprocessing function scales data to [0, 1) and randomly flips."""
    img = resize_op(d['image'])
    if config.data.random_flip:
        img = tf.image.random_flip_left_right(img)

    return dict(image=img, label=d.get('label', None))


def resize_op(img):
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)


def create_dataset(dataset_builder, split, count_repeat):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
        dataset_builder.download_and_prepare()
        ds = dataset_builder.as_dataset(
            split=split, shuffle_files=True, read_config=read_config)
    else:
        ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=count_repeat)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn)
    for batch_size in reversed(batch_dims):
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

train_ds = create_dataset(dataset_builder, train_split_name, num_epochs)
test_ds = create_dataset(dataset_builder, eval_split_name, 1)

"""# Classifier"""

classes = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')



## 6. Metric computation
@jax.jit
def compute_metrics(*, state, batch, rng):
    # logits = state.apply_fn({'params': state.params}, batch['image'])
    data_ = batch['image']
    data = jax.tree_map(lambda x: scaler(x), data_)
    ve_noise_scale = get_noise(data, rng)
    logits = state.apply_fn({'params': state.params}, data, ve_noise_scale)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""

    input_shape = (batch_size, 32, 32, 3)
    initial_params = module.init({'params': rng, 'dropout': jax.random.PRNGKey(0)},
                                 jnp.ones(input_shape, dtype=jnp.float32),
                                 jnp.ones((batch_size,), dtype=jnp.float32), train=True)

    model_state, classifier_params = initial_params.pop('params')

    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=classifier_params, tx=tx,
        metrics=Metrics.empty())


cnn_state = create_train_state(cnn, cnn_state_rng, learning_rate, momentum)

## 10. Train and evaluate


# since train_ds is replicated num_epochs times in get_datasets(), we divide by num_epochs
num_train_steps = train_ds.cardinality().numpy()
num_steps_per_epoch = num_train_steps // num_epochs
print(f"num_train_steps {num_train_steps}")
print(f"num_steps_per_epoch {num_steps_per_epoch}")
def get_noise(data, rng):
    rng, step_rng = random.split(rng)
    t = random.uniform(step_rng, (data.shape[0],), minval=sampling_eps, maxval=sde.T)
    ve_noise_scale = sde.marginal_prob(data, t)[1]

    return ve_noise_scale


@jax.jit
def train_step(state, batch, rng):
    """Train for a single step."""

    def loss_fn(params, rng):

        data_ = batch['image']
        data = jax.tree_map(lambda x: scaler(x), data_)
        ve_noise_scale = get_noise(data, rng)
        logits = state.apply_fn({'params': params}, data, ve_noise_scale)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params, rng)
    state = state.apply_gradients(grads=grads)
    return state


metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'test_loss': [],
                   'test_accuracy': []}


@jax.jit
def pred_step(state, batch):
    data_ = batch['image']
    data = jax.tree_map(lambda x: scaler(x), data_)


    ve_noise_scale = get_noise(data, rng)
    logits = state.apply_fn({'params': state.params}, data, ve_noise_scale)
    return logits.argmax(axis=1)

#train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
# eval_iter = iter(eval_ds)
# for step, batch in enumerate(train_ds.as_numpy_iterator()):

#for step in range(0, num_train_steps + 1):
for step, batch in enumerate(train_ds.as_numpy_iterator()):


    # Run optimization steps over training batches and compute batch metrics
    cnn_state = train_step(cnn_state, batch, rng)  # get updated train state (which contains the updated parameters)
    cnn_state = compute_metrics(state=cnn_state, batch=batch, rng=rng)  # aggregate batch metrics

    if (step + 1) % num_steps_per_epoch == 0:  # one training epoch has passed
        for metric, value in cnn_state.metrics.compute().items():  # compute metrics
            metrics_history[f'train_{metric}'].append(value)  # record metrics
        cnn_state = cnn_state.replace(metrics=cnn_state.metrics.empty())  # reset train_metrics for next training epoch

        # Compute metrics on the test set after each training epoch
        test_state = cnn_state
        for test_batch in test_ds.as_numpy_iterator():
            test_state = compute_metrics(state=test_state, batch=test_batch, rng=rng)

        for metric, value in test_state.metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)

        epoch = (step + 1) // num_steps_per_epoch
        current_test_accuracy = metrics_history['test_accuracy'][-1] * 100

        print(f"train epoch: {epoch}, "
              f"loss: {metrics_history['train_loss'][-1]}, "
              f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        print(f"test epoch: {epoch}, "
              f"loss: {metrics_history['test_loss'][-1]}, "
              f"accuracy: {current_test_accuracy}")

        logger.info(f"train epoch: {epoch}, "
                    f"loss: {metrics_history['train_loss'][-1]}, "
                    f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        logger.info(f"test epoch: {epoch}, "
                    f"loss: {metrics_history['test_loss'][-1]}, "
                    f"accuracy: {current_test_accuracy}")

        save_checkpoint_dict(cnn_state.params, epoch)

        test_batch = test_ds.as_numpy_iterator().next()
        pred = pred_step(cnn_state, test_batch)

        fig, axs = plt.subplots(5, 5, figsize=(12, 12))
        for i, ax in enumerate(axs.flatten()):
            # print(test_batch['image'].shape)
            ax.imshow(test_batch['image'][i])
            ax.set_title(f"label={pred[i]}, {classes[pred[i]]}")
            ax.axis('off')

        plt.savefig(f"{root_path}/samples/classifier/cnn_{epoch}.png")
        matplotlib.pyplot.close()
