import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB
import random


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self._lstm = tnn.LSTM(50, 100, batch_first=True, bidirectional=True, dropout=0.15, num_layers=2)
        self._conv = tnn.Conv1d(in_channels=200, out_channels=200, kernel_size=8, padding=5)
        self._pool = tnn.MaxPool1d(kernel_size=4)
        self._l1 = tnn.Linear(100,64)
        self._l2 = tnn.Linear(64,1)
        self._drop = tnn.Dropout()
        self._global = tnn.AdaptiveAvgPool1d(1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
        hidden = None
        out,h = self._lstm(input, hidden)
        out = torch.nn.functional.relu(self._conv(out.permute(0, 2, 1)))
        out = self._pool(out)
        out = torch.nn.functional.relu(self._conv(out))
        out = self._pool(out)
        out = torch.nn.functional.relu(self._conv(out))
        out = self._drop(out)
        out = self._global(out)
        out = torch.nn.functional.relu(self._l1(out.view(-1, 100)))
        out = self._l2(out)
        out = out.view(len(length), -1)
        return out[:, -1]

class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        newList = []
        for el in x:
            if "<" in el or ">" in el or el in ["the", "a", "and", "of", "to", "is", "in", "i", "-", "this", "that"]:
                continue
            newList.append(el)
        return newList

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch
    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")
    examples = []
    fields = [('text', textField), ('label', labelField)]
    wordDict = dict()
    for i in train:
        delText1 = []
        delText2 = []
        swapText = i.text.copy()
        for word in i.text:
            if random.random() > 0.5:
                delText1.append(word)
                if i.label in wordDict and len(wordDict[i.label]) > 0:
                    if random.random() > 0.5:
                        delText1.append(wordDict[i.label][random.randrange(0, len(wordDict[i.label]))])
            if random.random() > 0.5:
                delText2.append(word)
            if i.label in wordDict:
                wordDict[i.label].append(word)
            else:
                wordDict[i.label] = [word]
        index = 0
        while index < len(swapText)-1:
            if random.random() > 0.5:
                swapText[index], swapText[index+1] = swapText[index+1], swapText[index]
            index += 1
        index = 0
        while index < len(delText2)-1:
            if random.random() > 0.75:
                delText2[index], delText2[index+1] = delText2[index+1], delText2[index]                
            index += 1
        examples.append(data.Example.fromlist([delText1, i.label], fields))
        examples.append(data.Example.fromlist([delText2, i.label], fields))
        examples.append(data.Example.fromlist([swapText, i.label], fields))        
    extraData = data.Dataset(examples, fields)

    textField.build_vocab(train, dev, extraData, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev, extraData)

    trainLoader, testLoader, extraDataLoader = data.BucketIterator.splits((train, dev, extraData), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(20):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0
        
        for i, batch in enumerate(extraDataLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
