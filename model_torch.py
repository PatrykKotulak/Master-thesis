
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden)


def simple_tokinizer(text):
    """ Simple tokenizer
    """
    return text.split()


def precision(outputs, labels):
    op = outputs.cpu()
    la = labels.cpu()
    _, preds = torch.max(op, dim=1)
    return torch.tensor(precision_score(la, preds, average='weighted', zero_division=0))

def binary_precision(outputs, labels):
    la = labels.cpu()
    preds = torch.round(torch.sigmoid(outputs))
    preds = preds.detach().cpu().numpy()
    return torch.tensor(precision_score(la, preds, average='weighted', zero_division=0))


def accuracy_result(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    _, preds = torch.max(preds, 1)
    y_pred = preds.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    acc = accuracy_score(y, y_pred)
    return acc

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = accuracy_result(predictions, batch.label)

        prec = precision(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_prec += prec.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = accuracy_result(predictions, batch.label)

            prec = precision(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_prec += prec.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator)


def train_binary(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label.float())

        acc = binary_accuracy(predictions, batch.label)

        prec = binary_precision(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_prec += prec.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator)


def evaluate_binary(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label.float())

            acc = binary_accuracy(predictions, batch.label)

            prec = binary_precision(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_prec += prec.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
