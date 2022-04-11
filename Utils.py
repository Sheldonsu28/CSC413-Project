import numpy as np
import os
from torch.autograd import Variable
import cv2 as cv
import torch
import imageio


def calc_output_size(H, kernel_size, padding=0, dilation=1, stride=1):
    return ((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1


# Create annotation file
def build_index_file(dir="DIV2K_train_HR"):
    file_names = [[filename] for filename in os.listdir(dir)]

    np.savetxt(dir + "/" + dir + ".csv",
               file_names,
               delimiter=", ",
               fmt='% s',
               encoding="utf-8")


def crop_images(w, h, dir="DIV2K_train_HR"):
    file_names = [filename for filename in os.listdir(dir) if '.csv' not in filename]
    folder_hr = 'hr'
    for file in file_names:
        image = cv.imread(os.path.join(dir, file))
        img_size = image.shape
        x = img_size[1] / 2 - w / 2
        y = img_size[0] / 2 - h / 2
        crop_img = image[int(y):int(y + h), int(x):int(x + w)]
        cv.imwrite(os.path.join(folder_hr, file), crop_img)
    build_index_file(dir="hr")


def to_var(tensor, device):
    """Wraps a Tensor in a Variable, optionally placing it on the GPU.

        Arguments:
            tensor: A Tensor object.
            cuda: A boolean flag indicating whether to use the GPU.

        Returns:
            A Variable object, on the GPU if cuda==True.
    """

    return Variable(tensor.float()).cuda(device)


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape
    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w, :] = array[i * ncols + j].transpose(1, 2,
                                                                                                                 0)

    if channels == 1:
        result = result.squeeze()
    return result


def gan_save_samples(data, iteration, opts):
    generated_images = to_data(data)

    grid = create_image_grid(generated_images)

    # merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}.png'.format(iteration))
    imageio.imwrite(path, grid)
    print('Saved {}'.format(path))


if __name__ == "__main__":
    # a = calc_output_size(256, 9, stride=1, padding=4)
    # b = calc_output_size(a, 5, stride=1, padding=2)
    # print(calc_output_size(b, 5, stride=1, padding=2))
    crop_images(128, 128)
