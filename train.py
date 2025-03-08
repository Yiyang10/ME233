import argparse
import torch
import torch.optim as optim
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import Modeldiscover, loss
import utils

PATH_CHECKPOINT = os.path.join('trained_models', 'cp-{epoch:03d}.pt')
DIR_MODEL = 'trained_models'

LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
BATCH_SIZE = 8
WEIGHT_DECAY = 1e-6 # modifying the strength of L2 regularization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


def train_model(model, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        strain, targets = batch
        outputs = model(strain)

        targets = targets.reshape(-1, 1)
        loss_value = loss(targets, outputs)
        loss_value.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss_value.item(), epoch * len(train_loader) + batch_idx)

def test_model(model, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    num_batches = 0
    for batch in test_loader:
        strain, targets = batch
        strain.to(device)
        outputs = model(strain)
        test_loss += loss(targets.reshape(-1, 1), outputs).item()
        num_batches += 1

    test_loss /= num_batches
    writer.add_scalar('Loss/test', test_loss, epoch)
    return test_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select a mode: shear, tension, or compression.")

    # Create a mutually exclusive group so that only one mode can be selected at a time
    group = parser.add_mutually_exclusive_group()

    group.add_argument("--shear", dest="mode", action="store_const",
                       const="shear", help="Set mode to shear")
    group.add_argument("--t", dest="mode", action="store_const",
                       const="tension", help="Set mode to T")
    group.add_argument("--c", dest="mode", action="store_const",
                       const="compression", help="Set mode to C")

    # Set the default mode if none of the options are specified
    parser.set_defaults(mode="tension")

    # Parse the command-line arguments
    args = parser.parse_args()

    data_mode = args.mode # tension by default
    # Create directory if it doesn't exist
    os.makedirs(DIR_MODEL, exist_ok=True)

    # Load dataset
    #dataset_path = os.path.join('CANNsBRAINdata.xlsx')
    dataset_path = os.path.join('Ecoflex20.xlsx')
    train_loader, test_loader, _, _ = utils.dataloader(dataset_path, data_mode, batch_size = BATCH_SIZE, split_method="sequential")

    # Build model
    model = Modeldiscover(data_mode)
    if data_mode == "tension":
        path_model = os.path.join(DIR_MODEL, "trained_t.pt")
    elif data_mode == "compression":
        path_model = os.path.join(DIR_MODEL, "trained_c.pt")
    else:
        path_model = os.path.join(DIR_MODEL, "trained_shear.pt")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay= WEIGHT_DECAY)
    writer = SummaryWriter(log_dir='train_logs')

    patience = 1000
    best_loss = np.inf
    counter = 0

    # Train model
    for epoch in range(1, NUM_EPOCHS + 1):
        train_model(model, train_loader, optimizer, epoch, writer)
        loss_value = test_model(model, test_loader, writer, epoch)

        if loss_value < best_loss:
            best_loss = loss_value
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Validation Loss: {loss_value:.4f}, Counter: {counter}/{patience}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        # if epoch % 500 == 0:
        #     torch.save(model.state_dict(), PATH_CHECKPOINT.format(epoch=epoch))

    torch.save(best_model_state, path_model)
    writer.close()