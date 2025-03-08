import torch
import os
import numpy as np
from model import Modeldiscover
import utils
from train import BATCH_SIZE
import matplotlib.pyplot as plt
import argparse

DIR_CHECKPOINT = 'trained_models' + os.sep

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

    parser.set_defaults(mode="tension")
    args = parser.parse_args()
    data_mode = args.mode

    # Load dataset
    #dataset_path = os.path.join('CANNsBRAINdata.xlsx')
    dataset_path = os.path.join('Ecoflex20.xlsx')
    if data_mode == 'tension':
        file_name = "trained_t.pt"
        _, test_loader, stretch ,f_actual = utils.dataloader(dataset_path, "tension", batch_size=BATCH_SIZE)
    # Build model
        model = Modeldiscover(mode=data_mode)
        model.load_state_dict(torch.load(DIR_CHECKPOINT + 'trained_t.pt'))

    elif data_mode == 'compression':
        file_name = "trained_c.pt"
        _, test_loader, stretch, f_actual = utils.dataloader(dataset_path, "compression", batch_size=BATCH_SIZE)
    # Build model
        model = Modeldiscover(mode=data_mode)
        model.load_state_dict(torch.load(DIR_CHECKPOINT + 'trained_c.pt'))

    else:
        file_name = "trained_shear.pt"
        _, test_loader, stretch, f_actual = utils.dataloader(dataset_path, data_mode, batch_size=BATCH_SIZE)
        # Build model
        model = Modeldiscover(mode=data_mode)
        model.load_state_dict(torch.load(DIR_CHECKPOINT + 'trained_shear.pt'))
    model.eval()
    # a rough plot of the predictions

    # plt.plot(stretch, f_pred, color='blue', label='Prediction')
    # plt.plot(stretch, f_actual, color='red', label='Actual')
    # plt.legend()
    # plt.show()
    state_dict = torch.load(DIR_CHECKPOINT + file_name, map_location=torch.device('cpu'))

    # Print out the keys to see which parameters are stored.
    pack = utils.pack_up(state_dict)
    print(pack)
    utils.colored_plot(state_dict,stretch, f_actual, model, data_mode)
    # f_pred = model(stretch).detach().numpy()
    # plt.plot(stretch, f_pred)
    # plt.plot(stretch, f_actual)
    # print(stretch)
    # print(f_actual)
    # print(f_pred)
    #plt.show()

