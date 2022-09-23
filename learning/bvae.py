import torch
import unittest

from torchsummary import summary

from learning.beta_vae import BetaVAE


class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        self.model = BetaVAE(3, 10, loss_type='H').cuda()
        assert (torch.cuda.is_available())
        self.device = "cuda"

    def test_summary(self):
        print(summary(self.model, (3, 64, 64), device=self.device))
        # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64, device=self.device)
        y = self.model(x)
        print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64, device=self.device)
        result = self.model(x)
        loss = self.model.loss_function(*result, M_N = 0.005)
        print(loss)


if __name__ == '__main__':
    unittest.main()