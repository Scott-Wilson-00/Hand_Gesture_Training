import torch
from torch import nn
from torch.utils.data import DataLoader
from LandmarksDataset import HandLandmarksDataset
from NeuralNetwork import GestureNeuralNetwork

from torch.utils.tensorboard import SummaryWriter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_TO_INDEX = {"fistL": 0, "fistR": 1, "okayL": 2, "okayR": 3, "peaceL": 4, "peaceR": 5, "pointingL": 6, "pointingR": 7}

def train(dataloader, model, loss_fn, optimizer, summarywriter, epoch):
    size = len(dataloader.dataset)
    running_loss = 0
    model.train() #sets the model to training mode

    for batch, data in enumerate(dataloader):
        #--------------------------
        # Unpacking batch
        labels = data["labels"]
        landmarks = data["landmarks"]

        labels = [CLASS_TO_INDEX[label] for label in labels]
        labels = torch.tensor(labels, dtype=torch.long)
        
        labels = labels.to(DEVICE)
        landmarks = landmarks.to(DEVICE)
        #--------------------------
        # Compute prediction error
        pred = model(landmarks)
        loss = loss_fn(pred, labels)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #--------------------------
        running_loss += loss
        # if batch % 200 == 0:
        #     print("Running loss: " + str(running_loss.item()))

    print(f"Average Training Loss Epoch {epoch}: {running_loss/size}")
    
    summarywriter.add_scalars("Loss", {"train": running_loss/size}, epoch)


def test(dataloader, model, loss_fn, summarywriter, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            #--------------------------
            # Unpacking batch
            labels = data["labels"]
            landmarks = data["landmarks"]

            labels = [CLASS_TO_INDEX[label] for label in labels]
            labels = torch.tensor(labels, dtype=torch.long)
            
            labels = labels.to(DEVICE)
            landmarks = landmarks.to(DEVICE)
            #--------------------------
            # Calculating loss
            pred = model(landmarks)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    
    #--------------------------
    test_loss /= num_batches
    summarywriter.add_scalars("Loss", {"test": test_loss}, epoch)
    accuracy = (correct / size) * 100
    summarywriter.add_scalar("Accuracy/Testing", accuracy, epoch)
    print(f"Test Error: \n Accuracy: {accuracy}%, Avg loss: {test_loss} \n")



def train_and_test(epochs):
    #--------------------------
    # Setup
    model = GestureNeuralNetwork().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    train_dataset = HandLandmarksDataset('train_model/static/gesture_training_data.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = HandLandmarksDataset('train_model/static/gesture_testing_data.csv')
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    writer = SummaryWriter()

    #--------------------------
    # Training Loop
    for e in range(1, epochs+1):
        print(f"-------------- BEGIN TRAINING: EPOCH {e} --------------")
        train(train_dataloader, model, loss_fn, optimizer, writer, e)
        print(f"-------------- BEGIN TESTING: EPOCH {e} --------------")
        test(test_dataloader, model, loss_fn, writer, e)
    #--------------------------
    print("Done!")
    return model

def test_trained_model(path_to_model):

    model = GestureNeuralNetwork()
    model.load_state_dict(torch.load(path_to_model))

    test_dataset = HandLandmarksDataset('train_model/static/gesture_testing_data.csv')
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    writer = SummaryWriter()

    loss_fn = nn.CrossEntropyLoss()

    test(test_dataloader, model, loss_fn, writer, 0)


if __name__ == '__main__':
    model = train_and_test(15)
    # torch.save(model.state_dict(), "train_model/static/trained_model.pt")
    # print("Saved model!")

    # test_trained_model("train_model/static/trained_model.pt")


