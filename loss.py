import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from utils import wavelet_multi_mixer, calculate_local_variance, LocalMean, compute_gradients, build_pyramid

EPS = 1e-9
PI = 22.0 / 7.0


class LossZSXENet(nn.Module):
    """ZS-XENet total loss function.

    L_total = λ1 * L_PD_res + λ2 * L_PD_cons + λ3 * L_global +
              λ4 * L_local + λ5 * L_RD_res + λ6 * L_RD_cons + λ7 * L_mgf

    where:
      L_PD_res  -- PD-Net residual reconstruction loss (Eq. 6)
      L_PD_cons -- PD-Net consistency loss (Eq. 7)
      L_global  -- global illumination loss (Eq. 10)
      L_local   -- local illumination loss (Eq. 11)
      L_RD_res  -- RD-Net residual loss (Eq. 12)
      L_RD_cons -- RD-Net consistency loss (Eq. 13)
      L_mgf     -- Mutually Guided Filtering Loss (Eqs. 14-15)
    """

    def __init__(self):
        super(LossZSXENet, self).__init__()
        self._l2_loss = nn.MSELoss()
        self._l1_loss = nn.L1Loss()
        self.smooth_loss = SmoothLoss()
        self.texture_difference = TextureDifference()
        self.local_mean = LocalMean(patch_size=5)
        self.L_TV_loss = L_TV()
        self.L_mgf = MGFLLoss(alpha_t=0.02, alpha_r=0.02, num_scales=3)

    def forward(self, input, outs):
        """
        Args:
            input: original degraded X-ray image I
            outs: dictionary returned by ZSXENet.forward(input, num_pairs=N)
        """
        eps = 1e-9
        input = input + eps

        # --- Global tensors ---
        # I_tilde: preliminarily denoised image from PD-Net
        I_tilde = outs['I_tilde']
        # Il: estimated illumination map from ID-Net
        Il = outs['Il']
        # Ir: reflection component = I / Il
        Ir = outs['Ir']
        # Ir_refined: refined reflection from RD-Net
        Ir_refined = outs['Ir_refined']
        # Il_refined: refined illumination from RD-Net
        Il_refined = outs['Il_refined']
        # Blurred components for color consistency
        Ir_tilde_blur = outs['Ir_tilde_blur']
        Ir_refined_blur = outs['Ir_refined_blur']

        # --- Per-pair lists ---
        I_tilde1_list = outs['I_tilde1_list']
        I_tilde2_list = outs['I_tilde2_list']
        Il1_list = outs['Il1_list']
        Il2_list = outs['Il2_list']
        Ir1_list = outs['Ir1_list']
        Ir2_list = outs['Ir2_list']
        Ir1_tilde_list = outs['Ir1_tilde_list']
        Il1_tilde_list = outs['Il1_tilde_list']
        Ir2_tilde_list = outs['Ir2_tilde_list']
        Il2_tilde_list = outs['Il2_tilde_list']
        RD_out1_list = outs['RD_out1_list']
        RD_out2_list = outs['RD_out2_list']
        I_tilde1_I_tilde2_diff_list = outs['I_tilde1_I_tilde2_diff_list']
        Ir_d1_list = outs['Ir_d1_list']
        Ir_d2_list = outs['Ir_d2_list']
        Ir_d1_Ir_d2_diff_list = outs['Ir_d1_Ir_d2_diff_list']

        N = len(I_tilde1_list)
        total_loss = 0.0

        # ============ ID-Net losses: L_global + L_local ============
        # Compute enhancement factor α = Y_H / Y_L
        input_Y = I_tilde.detach()[:, 2, :, :] * 0.299 + I_tilde.detach()[:, 1, :, :] * 0.587 + I_tilde.detach()[:, 0, :, :] * 0.144
        input_Y_mean = torch.mean(input_Y, dim=(1, 2))
        enhancement_factor = 0.5 / (input_Y_mean + eps)  # α = Y_H / Y_L
        enhancement_factor = enhancement_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        enhancement_factor = torch.clamp(enhancement_factor, 1, 25)
        # Adaptive scaling factor β = α^(-1) * E^(-α)
        adjustment_ratio = torch.pow(0.9, -enhancement_factor) / enhancement_factor  # β
        adjustment_ratio = adjustment_ratio.repeat(1, 3, 1, 1)
        # Reflection from brightened image: I_tilde / Il
        normalized_low_light_layer = I_tilde.detach() / Il
        normalized_low_light_layer = torch.clamp(normalized_low_light_layer, eps, 0.8)
        # Target brightness: β * (α * I_tilde)^α
        enhanced_brightness = torch.pow(I_tilde.detach() * enhancement_factor, enhancement_factor)
        clamped_enhanced_brightness = torch.clamp(enhanced_brightness * adjustment_ratio, eps, 1)
        clamped_adjusted_low_light = torch.clamp(I_tilde.detach() * enhancement_factor, eps, 1)

        loss = 0
        # L_local (Eq. 11): local illumination loss -- pixel-wise (λ4=700)
        loss += self._l2_loss(Il, clamped_enhanced_brightness) * 700
        # L_global (Eq. 10): global illumination loss (λ3=1000)
        loss += self._l2_loss(normalized_low_light_layer, clamped_adjusted_low_light) * 1000
        # Total variation regularization on illumination map
        loss += self.L_TV_loss(Il) * 1600

        # ============ PD-Net and RD-Net pair-level losses ============
        # WMNG on input and I_tilde for pair-level constraints
        pairs_input = wavelet_multi_mixer(input, num_pairs=N)
        pairs_I_tilde = wavelet_multi_mixer(I_tilde, num_pairs=N)

        for n in range(N):
            I1_j, I2_j = pairs_input[n]
            denoised1, denoised2 = pairs_I_tilde[n]

            I_tilde1 = I_tilde1_list[n]
            I_tilde2 = I_tilde2_list[n]
            Il1_j = Il1_list[n]
            Il2_j = Il2_list[n]
            Ir1_j = Ir1_list[n]
            Ir2_j = Ir2_list[n]
            Ir1_tilde = Ir1_tilde_list[n]
            Il1_tilde = Il1_tilde_list[n]
            Ir2_tilde = Ir2_tilde_list[n]
            Il2_tilde = Il2_tilde_list[n]
            RD_out1 = RD_out1_list[n]
            RD_out2 = RD_out2_list[n]
            diff_I = I_tilde1_I_tilde2_diff_list[n]
            Ir_d1 = Ir_d1_list[n]
            Ir_d2 = Ir_d2_list[n]
            diff_Ir = Ir_d1_Ir_d2_diff_list[n]

            # ---------- L_PD_res (Eq. 6): PD-Net residual reconstruction loss (λ1=1000) ----------
            loss += self._l2_loss(I_tilde1, I2_j.detach()) * 1000
            loss += self._l2_loss(I_tilde2, I1_j.detach()) * 1000
            loss += self._l2_loss(I_tilde1, I_tilde2) * 1000
            # L_PD_cons (Eq. 7): PD-Net consistency loss (λ2=1000)
            loss += self._l2_loss(I_tilde1, denoised1) * 1000
            loss += self._l2_loss(I_tilde2, denoised2) * 1000

            # ---------- L_RD_res (Eq. 12): RD-Net residual loss (λ5=1000) ----------
            loss += self._l2_loss(RD_out1, torch.cat([Ir2_j.detach(), Il2_j.detach()], 1)) * 1000
            loss += self._l2_loss(RD_out2, torch.cat([Ir1_j.detach(), Il1_j.detach()], 1)) * 1000
            # L_RD_cons (Eq. 13): RD-Net consistency loss (λ6=1000)
            loss += self._l2_loss(RD_out1[:, 0:3, :, :], Ir_d1) * 1000
            loss += self._l2_loss(RD_out2[:, 0:3, :, :], Ir_d2) * 1000

            # Texture-guided consistency (auxiliary, commented out)
            # local_mean1 = self.local_mean(Ir_d1)
            # local_mean2 = self.local_mean(Ir_d2)
            # weighted_diff1 = (1 - diff_Ir) * local_mean1 + Ir_d1 * diff_Ir
            # weighted_diff2 = (1 - diff_Ir) * local_mean2 + Ir_d2 * diff_Ir

            # ---------- L_mgf (Eqs. 14-15): Mutually Guided Filtering Loss (λ7=10000) ----------
            loss_mgf = self.L_mgf(Ir_d1, Ir_d2)
            loss += loss_mgf * 10000

        loss /= N  # Average over N pairs

        # ============ Auxiliary global constraints ============
        # Color consistency (texture-guided)
        loss += self._l2_loss(Ir_tilde_blur.detach(), Ir_refined_blur) * 10000
        # Illumination consistency between ID-Net and RD-Net
        loss += self._l2_loss(Il.detach(), Il_refined) * 1000
        # Variance-based noise suppression
        noise_std = calculate_local_variance(Ir_refined - Ir)
        Ir_var = calculate_local_variance(Ir)
        loss += self._l2_loss(Ir_var, noise_std) * 1000

        return loss


# ===========================================================================
# Helper modules
# ===========================================================================

def local_mean(image):
    """Deprecated: use LocalMean class instead."""
    padding = self.patch_size // 2
    image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
    patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
    return patches.mean(dim=(4, 5))


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)

    return out_filter


class TextureDifference(nn.Module):
    """Computes texture difference between two images for structure-guided constraints."""

    def __init__(self, patch_size=5, constant_C=1e-5, threshold=0.975):
        super(TextureDifference, self).__init__()
        self.patch_size = patch_size
        self.constant_C = constant_C
        self.threshold = threshold

    def forward(self, image1, image2):
        # Convert RGB images to grayscale
        image1 = self.rgb_to_gray(image1)
        image2 = self.rgb_to_gray(image2)

        stddev1 = self.local_stddev(image1)
        stddev2 = self.local_stddev(image2)
        numerator = 2 * stddev1 * stddev2
        denominator = stddev1 ** 2 + stddev2 ** 2 + self.constant_C
        diff = numerator / denominator

        # Apply threshold to diff tensor
        binary_diff = torch.where(diff > self.threshold, torch.tensor(1.0, device=diff.device),
                                  torch.tensor(0.0, device=diff.device))

        return binary_diff

    def local_stddev(self, image):
        padding = self.patch_size // 2
        image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        mean = patches.mean(dim=(4, 5), keepdim=True)
        squared_diff = (patches - mean) ** 2
        local_variance = squared_diff.mean(dim=(4, 5))
        local_stddev = torch.sqrt(local_variance + 1e-9)
        return local_stddev

    def rgb_to_gray(self, image):
        # Convert RGB image to grayscale using the luminance formula
        gray_image = 0.144 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.299 * image[:, 2, :, :]
        return gray_image.unsqueeze(1)  # Add a channel dimension for compatibility


class L_TV(nn.Module):
    """Total variation loss for illumination smoothness."""

    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Blur(nn.Module):
    def __init__(self, nc):
        super(Blur, self).__init__()
        self.nc = nc
        kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1).cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(
                "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.nc))

        x = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)
        return x


class SmoothLoss(nn.Module):
    """Smoothness loss (currently unused in the final configuration)."""

    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p, dim=1,
                                        keepdim=True)
        pixel_grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p, dim=1,
                                        keepdim=True)
        pixel_grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p, dim=1,
                                        keepdim=True)
        pixel_grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p, dim=1,
                                        keepdim=True)
        pixel_grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p, dim=1,
                                        keepdim=True)
        pixel_grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p, dim=1,
                                        keepdim=True)
        pixel_grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p, dim=1,
                                        keepdim=True)
        pixel_grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p, dim=1,
                                        keepdim=True)
        pixel_grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p, dim=1,
                                        keepdim=True)
        pixel_grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p, dim=1,
                                        keepdim=True)
        pixel_grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p, dim=1,
                                        keepdim=True)
        pixel_grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p, dim=1,
                                        keepdim=True)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)

        total_term = ReguTerm1
        return total_term


class MGFLLoss(nn.Module):
    """Mutually Guided Filtering Loss (MGFL) — Eqs. 14-15.

    Enforces cross-noise structural consistency at the reflection-component level.
    Uses bidirectional gradient-domain consistency to highlight shared structural
    information (weld boundaries, defect contours) while suppressing noise.

    Based on mutually guided image filtering with sqrt-normalized gradient
    consistency, computed across multiple scales.
    """

    def __init__(self, alpha_t=0.02, alpha_r=0.02, eps_t=1e-2, eps_r=1e-2,
                 num_scales=3, scale_weight_decay=0.5):
        super().__init__()
        self.alpha_t = alpha_t
        self.alpha_r = alpha_r
        self.eps_t = eps_t
        self.eps_r = eps_r
        self.num_scales = num_scales
        self.scale_weight_decay = scale_weight_decay

    def surrogate_term(self, T, R):
        """Compute the gradient-domain energy term R(x1, x2) from Eq. 14.

        For each direction d in {h, v}:
            (∇_d T)^2 / (sqrt((∇_d R)^2 + ε_r^2) * sqrt((∇_d T)^2 + ε_t^2))
        """
        Th, Tv = compute_gradients(T)
        Rh, Rv = compute_gradients(R)

        denom_h = torch.sqrt(Rh.abs() ** 2 + self.eps_r ** 2) * torch.sqrt(Th.abs() ** 2 + self.eps_t ** 2)
        denom_v = torch.sqrt(Rv.abs() ** 2 + self.eps_r ** 2) * torch.sqrt(Tv.abs() ** 2 + self.eps_t ** 2)

        term_h = (Th ** 2) / denom_h
        term_v = (Tv ** 2) / denom_v
        return (term_h + term_v).mean()

    def forward(self, T, R):
        """Compute L_mgf in bidirectional form (Eq. 15).

        L_mgf(T, R) = λ1 * R(T, R) + λ2 * R(R, T)

        Multi-scale: computed on average-pooled pyramid levels.
        T = Ir1_j, R = Ir2_j (reflection components from different noise perturbations).
        """
        pyr_T = build_pyramid(T, self.num_scales)
        pyr_R = build_pyramid(R, self.num_scales)

        total_loss = 0.0
        weight = 1.0
        for s in range(self.num_scales):
            T_s, R_s = pyr_T[s], pyr_R[s]
            loss_scale = (
                    self.alpha_t * self.surrogate_term(T_s, R_s) +
                    self.alpha_r * self.surrogate_term(R_s, T_s)
            )
            total_loss += weight * loss_scale
            weight *= self.scale_weight_decay

        return total_loss
