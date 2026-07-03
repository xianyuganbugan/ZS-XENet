import torch
import torch.nn as nn
from loss import LossZSXENet, TextureDifference
from utils import blur, wavelet_multi_mixer  # WMNG: Wavelet-based Multi-noise Generation


class PDNet(nn.Module):

    def __init__(self, chan_embed=48):
        super(PDNet, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(3, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 3, 1)

    def forward(self, x):
        # f_PD(x) = Conv3x3(σ(Conv3x3(σ(Conv3x3(x)))))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class RDNet(nn.Module):

    def __init__(self, chan_embed=96):
        super(RDNet, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(6, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 6, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class IDNet(nn.Module):

    def __init__(self, layers, channels):
        super(IDNet, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        fea = torch.clamp(fea, 0.0001, 1)

        return fea


class ZSXENet(nn.Module):

    def __init__(self):
        super(ZSXENet, self).__init__()

        self.id_net = IDNet(layers=3, channels=64)
        self.pd_net = PDNet(chan_embed=48)
        self.rd_net = RDNet(chan_embed=48)
        self._l2_loss = nn.MSELoss()
        self._l1_loss = nn.L1Loss()
        self._criterion = LossZSXENet()
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.TextureDifference = TextureDifference()

    def id_net_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def pd_net_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input, num_pairs=4):
        eps = 1e-4
        input = input + eps
        # WMNG: generate n noise-variant image pairs
        pairs = wavelet_multi_mixer(input, num_pairs=num_pairs)
        # PD-Net: preliminary denoising of full image
        I_tilde = input - self.pd_net(input)
        I_tilde = torch.clamp(I_tilde, eps, 1)
        # ID-Net: estimate illumination map Il from I_tilde
        Il = self.id_net(I_tilde.detach())
        # WMNG on illumination map to generate noise-diverse variants
        Il_pairs = wavelet_multi_mixer(Il, num_pairs=num_pairs)

        # Store per-pair variables
        I_tilde1_list = []
        I_tilde2_list = []
        Ir11_list = []
        Ir12_list = []
        RD_out1_list = []
        RD_out2_list = []
        Ir1_tilde_list = []
        Il1_tilde_list = []
        Ir2_tilde_list = []
        Il2_tilde_list = []
        I_tilde1_I_tilde2_diff_list = []

        for n, (I1_j, I2_j) in enumerate(pairs):
            # PD-Net: predict noise and subtract
            I_tilde1 = I1_j - self.pd_net(I1_j)
            I_tilde1 = torch.clamp(I_tilde1, eps, 1)
            I_tilde2 = I2_j - self.pd_net(I2_j)
            I_tilde2 = torch.clamp(I_tilde2, eps, 1)
            I_tilde1_list.append(I_tilde1)
            I_tilde2_list.append(I_tilde2)

            Il1_j, Il2_j = Il_pairs[n]

            # Reflection component: Ir = I / Il
            Ir1_j = I1_j / (Il1_j + eps)
            Ir1_j = torch.clamp(Ir1_j, 0, 1)
            Ir2_j = I2_j / (Il2_j + eps)
            Ir2_j = torch.clamp(Ir2_j, 0, 1)
            Ir11_list.append(Ir1_j)
            Ir12_list.append(Ir2_j)

            # RD-Net: refine reflection + illumination
            cat1 = torch.cat([Ir1_j, Il1_j], dim=1)
            RD_out1 = cat1.detach() - self.rd_net(cat1)
            RD_out1 = torch.clamp(RD_out1, eps, 1)
            RD_out1_list.append(RD_out1)
            Ir1_tilde = RD_out1[:, :3, :, :]
            Il1_tilde = RD_out1[:, 3:, :, :]
            Ir1_tilde_list.append(Ir1_tilde)
            Il1_tilde_list.append(Il1_tilde)

            # RD-Net for the second pair
            cat2 = torch.cat([Ir2_j, Il2_j], dim=1)
            RD_out2 = cat2.detach() - self.rd_net(cat2)
            RD_out2 = torch.clamp(RD_out2, eps, 1)
            RD_out2_list.append(RD_out2)
            Ir2_tilde = RD_out2[:, :3, :, :]
            Il2_tilde = RD_out2[:, 3:, :, :]
            Ir2_tilde_list.append(Ir2_tilde)
            Il2_tilde_list.append(Il2_tilde)

            I_tilde1_I_tilde2_diff_list.append(self.TextureDifference(I_tilde1, I_tilde2))

        # Reflection from full image decomposition
        Ir = input / Il
        Ir = torch.clamp(Ir, eps, 1)

        # RD-Net on full image reflection + illumination
        H5_pred = torch.cat([Ir, Il], dim=1).detach() - self.rd_net(torch.cat([Ir, Il], dim=1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        Ir_refined = H5_pred[:, :3, :, :]
        Il_refined = H5_pred[:, 3:, :, :]

        # WMNG on refined reflection for consistency constraints
        Ir_pairs = wavelet_multi_mixer(Ir_refined, num_pairs=num_pairs)
        Ir_d1_list = []
        Ir_d2_list = []
        Ir_d1_Ir_d2_diff_list = []
        for (Ir_d1, Ir_d2) in Ir_pairs:
            Ir_d1_list.append(Ir_d1)
            Ir_d2_list.append(Ir_d2)
            Ir_d1_Ir_d2_diff_list.append(self.TextureDifference(Ir_d1, Ir_d2))

        # Reflection from I_tilde decomposition (for blur loss)
        Ir_tilde = I_tilde / Il
        Ir_tilde = torch.clamp(Ir_tilde, 0, 1)
        Ir_tilde_blur = blur(Ir_tilde)
        Ir_refined_blur = blur(Ir_refined)

        def _stack_if_list(lst):
            return torch.stack(lst, dim=0) if len(lst) > 0 else None

        outputs = {
            # PD-Net outputs: preliminarily denoised results
            'I_tilde1_list': I_tilde1_list,
            'I_tilde2_list': I_tilde2_list,
            'I_tilde1_stack': _stack_if_list(I_tilde1_list),
            'I_tilde2_stack': _stack_if_list(I_tilde2_list),
            # ID-Net outputs: illumination variant pairs
            'Il1_list': [p[0] for p in Il_pairs],
            'Il2_list': [p[1] for p in Il_pairs],
            # Reflection components
            'Ir1_list': Ir11_list,
            'Ir2_list': Ir12_list,
            # RD-Net outputs
            'RD_out1_list': RD_out1_list,
            'RD_out2_list': RD_out2_list,
            'Ir1_tilde_list': Ir1_tilde_list,
            'Il1_tilde_list': Il1_tilde_list,
            'Ir2_tilde_list': Ir2_tilde_list,
            'Il2_tilde_list': Il2_tilde_list,
            # Texture differences
            'I_tilde1_I_tilde2_diff_list': I_tilde1_I_tilde2_diff_list,
            'Ir_d1_list': Ir_d1_list,
            'Ir_d2_list': Ir_d2_list,
            'Ir_d1_Ir_d2_diff_list': Ir_d1_Ir_d2_diff_list,
            # Global tensors
            'I_tilde': I_tilde,
            'Il': Il,
            'Ir': Ir,
            'Ir_refined': Ir_refined,
            'Il_refined': Il_refined,
            'Ir_tilde_blur': Ir_tilde_blur,
            'Ir_refined_blur': Ir_refined_blur,
        }

        return outputs

    def _loss(self, input, num_pairs=4):
        outs = self.forward(input, num_pairs=num_pairs)
        loss = self._criterion(input, outs)

        return loss


class ZSXENetFinetune(nn.Module):
    def __init__(self, weights):
        super(ZSXENetFinetune, self).__init__()

        self.id_net = IDNet(layers=3, channels=64)
        self.pd_net = PDNet(chan_embed=48)
        self.rd_net = RDNet(chan_embed=48)

        base_weights = torch.load(weights, map_location='cuda:0')
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        eps = 1e-4
        input = input + eps
        # PD-Net: preliminary denoising
        I_tilde = input - self.pd_net(input)
        I_tilde = torch.clamp(I_tilde, eps, 1)
        # ID-Net: estimate illumination map
        Il = self.id_net(I_tilde)
        # Reflection decomposition
        Ir = input / Il
        Ir = torch.clamp(Ir, eps, 1)
        # RD-Net: refine reflection + illumination
        H5_pred = torch.cat([Ir, Il], 1).detach() - self.rd_net(torch.cat([Ir, Il], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        Ir_refined = H5_pred[:, :3, :, :]
        return Ir, Ir_refined
