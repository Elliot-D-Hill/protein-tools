import time

from torch import int64, tensor, cat, no_grad, save
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset

from utils import (
    get_tokens,
    make_dataset_iterator,
    print_batch_info,
    print_epoch_info,
    split_dataset,
)


class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(SequenceClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class EmbeddingModel:
    def __init__(
        self, X, y, vocabulary, params, model, criterion, optimizer, scheduler, device
    ):
        self.device = device
        # dataset
        self.X = X
        self.y = y
        self.label_map = {value: key for key, value in enumerate(set(y))}
        self.vocabulary = vocabulary
        # model hyperparameters
        self.embedding_size = params.embedding_size
        # training hyperparameters
        self.batch_size = params.batch_size
        self.learning_rate = params.learning_rate
        self.n_epochs = params.n_epochs
        self.train_proportion = params.train_proportion
        self.clip_grad = params.clip_grad
        self.gamma = params.lr_scheduler_gamma
        # model objects
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        # helper variables
        self.label_type = int64

    def get_label(self, _label):
        return self.label_map[_label]

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.get_label(_label))
            processed_text = tensor(get_tokens(_text, self.vocabulary), dtype=int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = tensor(label_list, dtype=self.label_type).float().unsqueeze(1)
        offsets = tensor(offsets[:-1]).cumsum(dim=0)
        text_list = cat(text_list)
        return (
            label_list.to(self.device),
            text_list.to(self.device),
            offsets.to(self.device),
        )

    def make_dataloader(self, X, y):
        data_iterator = make_dataset_iterator(X, y)
        dataset_map = to_map_style_dataset(data_iterator)
        return DataLoader(
            dataset_map,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
        )

    # FIXME does not use self, turn into a function
    def calculate_metric(self, predicted_label, label, total_acc, total_count):
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        return total_acc, total_count

    def train_epoch(self, dataloader, epoch, log_interval=10):
        total_acc, total_count = 0, 0
        n_batches = len(dataloader)
        for idx, (label, text, offsets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            predicted_label = self.model(text, offsets)
            loss = self.criterion(predicted_label, label)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            total_acc, total_count = self.calculate_metric(
                predicted_label, label, total_acc, total_count
            )
            if idx % log_interval == 0 and idx > 0:
                accuracy = total_acc / total_count
                print_batch_info(epoch, idx, n_batches, accuracy)
                total_acc, total_count = 0, 0

    def evaluate(self, dataloader):
        self.model.eval()
        total_acc, total_count = 0, 0
        with no_grad():
            for label, text, offsets in dataloader:
                predicted_label = self.model(text, offsets)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count

    def train_model(self, train_loader, validation_loader):
        self.model.train()
        total_accu = None
        for epoch in range(1, self.n_epochs + 1):
            epoch_start_time = time.time()
            self.train_epoch(train_loader, epoch)
            accu_val = self.evaluate(validation_loader)
            if total_accu is not None and total_accu > accu_val:
                self.scheduler.step()
            else:
                total_accu = accu_val
            print_epoch_info(epoch, epoch_start_time, accu_val)

    def save_model(self, PATH):
        save(self.model.state_dict(), PATH)

    def run_pipeline(self):
        splits = split_dataset(self.X, self.y, self.train_proportion)
        dataloaders = [self.make_dataloader(X, y) for X, y in splits]
        train, test, validation = dataloaders
        self.train_model(train, validation)
        test_accuracy = self.evaluate(test)
        return test_accuracy
