import argparse
import torch
from torch import nn, optim
import futility
import fmodel

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network for flower classification')
    parser.add_argument('data_dir', action="store", help='Directory for the dataset', default="./flowers/")
    parser.add_argument('--save_dir', action="store", help='Directory to save the trained model checkpoint', default="./checkpoint.pth")
    parser.add_argument('--arch', action="store", help='Architecture for the neural network', default="vgg16")
    parser.add_argument('--learning_rate', action="store", type=float, help='Learning rate for training', default=0.001)
    parser.add_argument('--hidden_units', action="store", type=int, help='Number of hidden units', default=512)
    parser.add_argument('--epochs', action="store", type=int, help='Number of epochs for training', default=3)
    parser.add_argument('--dropout', action="store", type=float, help='Dropout rate', default=0.2)
    parser.add_argument('--gpu', action="store_true", help='Use GPU for training')

    return parser.parse_args()

def train_model(trainloader, validloader, model, criterion, optimizer, device, epochs=3, print_every=20):
    model.to(device)
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, accuracy = futility.validate_model(model, criterion, validloader, device)
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss:.3f}.. "
                      f"Validation accuracy: {accuracy:.3f}")
                running_loss = 0

def main():
    args = parse_arguments()
    where = args.data_dir
    path = args.save_dir
    lr = args.learning_rate
    struct = args.arch
    hidden_units = args.hidden_units
    power = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    epochs = args.epochs
    dropout = args.dropout

    trainloader, validloader, _, train_data = futility.load_data(where)
    model, criterion, optimizer = fmodel.setup_network(struct, dropout, hidden_units, lr, power)

    train_model(trainloader, validloader, model, criterion, optimizer, power, epochs)

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'structure': struct,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'learning_rate': lr,
        'no_of_epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, path)
    print(f"Model checkpoint saved to {path}")

if __name__ == "__main__":
    main()
