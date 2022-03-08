def calc_output_size(H, kernel_size, padding=0, dilation=1, stride=1):
    return ((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1


if __name__ == "__main__":
    a = 1080
    a = calc_output_size(a, 3, stride=1)
    print(a)
