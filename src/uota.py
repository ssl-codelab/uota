import torch
import torch.nn as nn
import torch.distributed as dist
from apex import amp

from .resnet50 import ResNet, BasicBlock, Bottleneck
from .utils import concat_all_gather


class UOTA(ResNet):
    """
    Build a UOTA model.
    """
    def __init__(self,
                 uota_tau=350.,
                 uota_eps=1e-8,
                 *args,
                 **kwargs):
        super(UOTA, self).__init__(*args, **kwargs)
        self.uota_tau = uota_tau
        self.uota_eps = uota_eps

    def _get_sample_weight(self, x, bs):
        with torch.no_grad():
            q_times = x.size(0) // bs  # q_times is M
            x_mu = x[:2 * bs, :].reshape(2, bs, -1).mean(dim=0).unsqueeze(0)  # refer to Eq.(M.10)
            x_delta = x.reshape(q_times, bs, -1) - x_mu
            x_delta_t = x_delta.reshape(q_times * bs, -1)
            x_delta = x_delta_t.permute(1, 0).contiguous()
            x_delta_cat_t = concat_all_gather(x_delta_t)
            x_delta_cat = x_delta_cat_t.permute(1, 0).contiguous()

            # compute sigma and sigma inverse
            sigma = torch.matmul(x_delta_cat, x_delta_cat_t) / float(x_delta_cat_t.size(0))
            eps_mat = self.uota_eps * torch.eye(sigma.size(-1), sigma.size(-1)).to(sigma.device)
            sigma += eps_mat
            sigma_inverse = torch.inverse(sigma)  # refer to Eq.(M.10)

            # compute uncertainty: refer to Eq.(M.9)
            sigma_inverse /= self.uota_tau
            tmp = torch.matmul(x_delta_t, sigma_inverse)
            uncertainty_mat = torch.matmul(tmp, x_delta)
            uncertainty_sim = uncertainty_mat.diagonal(dim1=-2, dim2=-1).contiguous()
            uncertainty_sim_cat = concat_all_gather(uncertainty_sim)

            # normalize
            sample_weight = (-uncertainty_sim_cat).softmax(dim=0)  # refer to Eq.(M.11)
            sample_weight *= float(q_times * bs * torch.distributed.get_world_size())  # maintain the loss energy by multiplying MN
            rank = dist.get_rank()
            sample_weight = sample_weight[rank*q_times*bs:(rank+1)*q_times*bs]
        return sample_weight

    def forward_head(self, x, bs):
        if self.projection_head is not None:
            x = self.projection_head(x)
        xtype = x.dtype
        if self.l2norm:
            with torch.no_grad():
                x_float32 = x.clone()
                x_float32 = x_float32.type(torch.float32)
                x_float32 = nn.functional.normalize(x_float32, dim=1, p=2)
            x = nn.functional.normalize(x, dim=1, p=2)

        # get sample weight
        if xtype == torch.float16:
            with amp.handle.disable_casts():
                sample_weight = self._get_sample_weight(x_float32, bs)
        else:
            sample_weight = self._get_sample_weight(x_float32, bs)

        if self.prototypes is not None:
            return x, self.prototypes(x), sample_weight
        return x, sample_weight

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True)[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output, inputs[0].size(0))


def uota_r18(**kwargs):
    return UOTA(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


def uota_r50(**kwargs):
    return UOTA(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)