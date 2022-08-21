from dataclasses import dataclass
from decimal import Decimal
import time

from sklearn.model_selection import train_test_split
from torch import mean, sum


# returns a tuble containing the sequence header and the sequence itself
def separate_header(sequence):
    return sequence[0], "".join(sequence[1:])


def build_vocabulary(text_list):
    unique_tokens = set("".join(text_list))
    return {value: key for key, value in enumerate(unique_tokens)}


def print_batch_info(metrics):
    text = ""
    for (
        name,
        value,
    ) in metrics.items():
        text += f"{name}: {value}, "
    return text


def print_batch_info(epoch, idx, n_batches, metric):
    text = f"| epoch {epoch} | {idx} / {n_batches} batches | metric: {metric:.5f} |"
    print(text)


def log_accuracy(
    total_acc, total_count, predicted_label, label, idx, log_interval, epoch, n_batches
):
    total_acc += (predicted_label.argmax(1) == label).sum().item()
    total_count += label.size(0)
    if idx % log_interval == 0 and idx > 0:
        print_batch_info(epoch, idx, n_batches, total_acc, total_count)
        total_acc, total_count = 0, 0
    return total_acc, total_count


def print_epoch_info(epoch, start_time, metrics):
    print("-" * 85)
    time_i = time.time() - start_time
    text = f"| end of epoch {epoch:3d} | time: {time_i:5.2f}s | validation metrics: {metrics} "
    print(text)
    print("-" * 85)


def make_dataset_iterator(X, y):
    return iter([(y_i, X_i) for y_i, X_i in zip(y, X)])


def get_tokens(text, vocabulary):
    return [vocabulary.get(key) for key in list(text)]


def split_dataset(X, y, train_proportion):
    test_valid_proportion = 1 - train_proportion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_valid_proportion, random_state=2
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=2
    )
    return [(X_train, y_train), (X_test, y_test), (X_val, y_val)]


def r2_loss(output, target):
    target_mean = mean(target)
    ss_tot = sum((target - target_mean) ** 2)
    ss_res = sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


@dataclass
class Hyperparameters:
    # learning hyperparameters
    learning_rate: Decimal
    batch_size: int
    n_epochs: int
    train_proportion: Decimal
    clip_grad: Decimal
    lr_scheduler_gamma: Decimal
    # model hyperparameters
    embedding_size: int
