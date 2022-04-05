import numpy as np
import os
from torch.autograd import Variable


def calc_output_size(H, kernel_size, padding=0, dilation=1, stride=1):
    return ((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1


# Create annotation file
def data_processing(dir="DIV2K_train_HR", scales=None):
    if scales is None:
        scales = [0.5, 0.25]

    file_names = [[filename] for filename in os.listdir(dir)]

    np.savetxt(dir + "/" + dir + ".csv",
               file_names,
               delimiter=", ",
               fmt='% s',
               encoding="utf-8")


def to_var(tensor, device):
    """Wraps a Tensor in a Variable, optionally placing it on the GPU.

        Arguments:
            tensor: A Tensor object.
            cuda: A boolean flag indicating whether to use the GPU.

        Returns:
            A Variable object, on the GPU if cuda==True.
    """

    return Variable(tensor.float()).cuda(device)


if __name__ == "__main__":
    a = calc_output_size(256, 9, stride=1, padding=4)
    b = calc_output_size(a, 5, stride=1, padding=2)
    print(calc_output_size(b, 5, stride=1, padding=2))
    # data_processing()
