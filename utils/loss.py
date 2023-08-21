import torch


def loss_sstfusion(I, I_tilde, epsilon=1e-3):
    # epsilon = torch.from_numpy(epsilon)
    return torch.sqrt(torch.norm(I - I_tilde, p=1) ** 2 + epsilon ** 2)


if __name__ == '__main__':
    i = torch.randn((1, 1, 256, 256))
    i_t = torch.randn((1, 1, 256, 256))

    print(loss_sstfusion(i, i_t))
