import torch
from torch import nn
from torch.utils.data import DataLoader
from LandmarksDataset import HandLandmarksDataset
from NeuralNetwork import GestureNeuralNetwork
from torch.utils.tensorboard import SummaryWriter
import os

CLASSES = ('peaceR', 'peaceL', 'fistR', 'fistL', 'pointingR', 'pointingL', 'okayR', 'okayL',)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

net = GestureNeuralNetwork().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

script_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = script_dir + '/gesture_training_data.csv'
test_file_path = script_dir + '/gesture_testing_data.csv'

if __name__ == '__main__':

    train_dataset = HandLandmarksDataset(train_file_path)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataset = HandLandmarksDataset(test_file_path)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    dataiter = iter(train_dataloader)
    batch = next(dataiter)
    labels = batch['labels']
    landmarks = batch['landmarks']
    labels_str = ', '.join(labels)
    
    writer = SummaryWriter()
    writer.add_graph(net, landmarks)
    writer.close()

