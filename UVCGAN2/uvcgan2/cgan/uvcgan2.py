# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import itertools
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torchvision.transforms import GaussianBlur, Resize

from uvcgan2.torch.select            import (
    select_optimizer, extract_name_kwargs
)
from uvcgan2.torch.queue             import FastQueue
from uvcgan2.torch.funcs             import prepare_model, update_average_model
from uvcgan2.torch.layers.batch_head import BatchHeadWrapper, get_batch_head
from uvcgan2.base.losses             import GANLoss
from uvcgan2.torch.gradient_penalty  import GradientPenalty
from uvcgan2.models.discriminator    import construct_discriminator
from uvcgan2.models.generator        import construct_generator
from torch import nn

from .model_base import ModelBase
from .named_dict import NamedDict
from .funcs import set_two_domain_input

import cv2
import numpy as np
import os
import torch.nn.functional as F
from torchvision.transforms import Compose
# from depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import random
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils
from typing import Any, Dict, List, Optional, Tuple, Union
from torchvision.utils import save_image

def save_tensor_as_image(tensor, file_path):
    """
    tensor: 이미지 데이터를 가진 텐서, (C, H, W) 또는 (N, C, H, W)의 차원을 가질 수 있습니다.
    file_path: 저장할 파일 경로와 파일 이름
    """
    # 텐서의 값 범위를 [0, 1]로 정규화합니다.
    # 이는 이미지 데이터가 일반적으로 0에서 1 또는 0에서 255 사이의 값을 가정하기 때문입니다.
    # 이미 텐서가 이 범위에 있지 않다면 이 단계를 조정해야 할 수도 있습니다.
    tensor = torch.clamp(tensor, 0, 1)
    
    # 이미지 저장
    save_image(tensor, file_path)

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def warp_flow_torch(img, flow):
    b, c, h, w = img.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(w, device=img.device), torch.arange(h, device=img.device), indexing='xy')
    flow = -flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    flow[..., 0] += grid_x.float()
    flow[..., 1] += grid_y.float()
    
    norm_x = 2.0 * flow[..., 0] / (w - 1) - 1.0
    norm_y = 2.0 * flow[..., 1] / (h - 1) - 1.0
    norm_flow = torch.stack([norm_x, norm_y], dim=-1)
    
    warped_img = F.grid_sample(img, norm_flow, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped_img

def prepare_tensor_inputs(
    adapter,
    images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    flows: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    image_only: bool = False,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """Transform tensor inputs into the input format of the optical flow models outside of the IOAdapter class.

    Parameters
    ----------
    adapter : IOAdapter
        An instance of IOAdapter which provides necessary methods like scaling and device management.
    images : Union[torch.Tensor, List[torch.Tensor]], optional
        One or more images as tensors. Typically, this will be a list with two images.
    flows : Union[torch.Tensor, List[torch.Tensor]], optional
        One or more optical flows as tensors, used for validation.
    image_only : bool, optional
        If True, only applies scaling and padding to the images.
    device : torch.device, optional
        The device tensors should be transferred to. If None, defaults to CPU.

    Returns
    -------
    Dict[str, torch.Tensor]
        The inputs converted and transformed to the input format of the optical flow models.
    """
    inputs = {}
    if images is not None:
        inputs['images'] = images
    if flows is not None:
        inputs['flows'] = flows

    for k, v in inputs.items():
        if isinstance(v, list):
            v = torch.stack(v).to(device if device is not None else 'cpu')  # Convert list of tensors to a single tensor and move to specified device
            v = v.permute(1, 0, 2, 3, 4)  # Permute dimensions to match (B, S, C, H, W)

        if image_only and k != 'images':
            continue

        while len(v.shape) < 5:  # Ensure tensor has 5 dimensions: (B, S, C, H, W)
            v = v.unsqueeze(0)
        
        if adapter.scaler is not None and (k == 'images' or k == 'flows'):
            v = adapter.scaler.fill(v, is_flow=k.startswith('flow'))
        
        inputs[k] = v

    return inputs

class MaxPool2dTransform(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(MaxPool2dTransform, self).__init__()
        # MaxPool2d layer를 초기화합니다. stride가 None일 경우, kernel_size 값을 사용합니다.
        self.maxpool = nn.MaxPool2d(kernel_size, stride if stride is not None else kernel_size, padding, dilation, ceil_mode)

    def forward(self, img):
        return self.maxpool(img)  # MaxPooling 적용
        



def construct_consistency_model(consist, device):
    name, kwargs = extract_name_kwargs(consist)
    maxpool_kwargs = {'kernel_size': 2, 'stride': 2, 'padding': 0, 'dilation': 1, 'ceil_mode': False}

    # print('kwargs')
    # print(kwargs)
    # print('kwargs-end')
    if name == 'blur':
        return GaussianBlur(**kwargs).to(device)

    if name == 'resize':
        return Resize(**kwargs).to(device)

    if name == 'maxpool':
        return MaxPool2dTransform(**maxpool_kwargs).to(device)

    
    raise ValueError(f'Unknown consistency type: {name}')

def queued_forward(batch_head_model, input_image, queue, update_queue = True):
    output, pred_body = batch_head_model.forward(
        input_image, extra_bodies = queue.query(), return_body = True
    )

    if update_queue:
        queue.push(pred_body)

    return output

def update_previous_results(self): ## By Grimoire
        """이전 배치 결과를 현재 배치 결과로 업데이트하는 함수"""
        if self.images.fake_a is not None:
            self.prev_fake_a = self.images.fake_a.detach().clone()
        if self.images.fake_b is not None:
            self.prev_fake_b = self.images.fake_b.detach().clone()
        if self.images.real_a is not None:
            self.prev_real_a = self.images.real_a.detach().clone()

class UVCGAN2(ModelBase):
    # pylint: disable=too-many-instance-attributes

    def _setup_images(self, _config):
        images = [
            'real_a', 'real_b',
            'fake_a', 'fake_b',
            'reco_a', 'reco_b',
            'consist_real_a', 'consist_real_b',
            'consist_fake_a', 'consist_fake_b',
            'prev_real_a', 'prev_real_b',
            'prev_fake_a', 'prev_fake_b',
            'warped_fake_b', 'fake_b_warped'
        ]

        if self.is_train and self.lambda_idt > 0:
            images += [ 'idt_a', 'idt_b', ]

        return NamedDict(*images)

    def _construct_batch_head_disc(self, model_config, input_shape):
        disc_body = construct_discriminator(
            model_config, input_shape, self.device
        )

        disc_head = get_batch_head(self.head_config)
        disc_head = prepare_model(disc_head, self.device)

        return BatchHeadWrapper(disc_body, disc_head)

    def _setup_models(self, config):
        models = {}

        shape_a = config.data.datasets[0].shape
        shape_b = config.data.datasets[1].shape

        models['gen_ab'] = construct_generator(
            config.generator, shape_a, shape_b, self.device
        )
        models['gen_ba'] = construct_generator(
            config.generator, shape_b, shape_a, self.device
        )

        if self.avg_momentum is not None:
            models['avg_gen_ab'] = construct_generator(
                config.generator, shape_a, shape_b, self.device
            )
            models['avg_gen_ba'] = construct_generator(
                config.generator, shape_b, shape_a, self.device
            )

            models['avg_gen_ab'].load_state_dict(models['gen_ab'].state_dict())
            models['avg_gen_ba'].load_state_dict(models['gen_ba'].state_dict())

        if self.is_train:
            models['disc_a'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[0].shape
            )
            models['disc_b'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[1].shape
            )

        return NamedDict(**models)

    def _setup_losses(self, config):
        losses = [
            'gen_ab', 'gen_ba', 'cycle_a', 'cycle_b', 'disc_a', 'disc_b', 'flow'
        ]

        if self.is_train and self.lambda_idt > 0:
            losses += [ 'idt_a', 'idt_b' ]

        if self.is_train and config.gradient_penalty is not None:
            losses += [ 'gp_a', 'gp_b' ]

        if self.consist_model is not None:
            losses += [ 'consist_a', 'consist_b' ]

        return NamedDict(*losses)

    def _setup_optimizers(self, config):
        optimizers = NamedDict('gen', 'disc')

        optimizers.gen = select_optimizer(
            itertools.chain(
                self.models.gen_ab.parameters(),
                self.models.gen_ba.parameters()
            ),
            config.generator.optimizer
        )

        optimizers.disc = select_optimizer(
            itertools.chain(
                self.models.disc_a.parameters(),
                self.models.disc_b.parameters()
            ),
            config.discriminator.optimizer
        )

        return optimizers

    def __init__(
        self, savedir, config, is_train, device, head_config = None,
        lambda_a        = 10.0,
        lambda_b        = 10.0,
        lambda_idt      = 0.5,
        lambda_consist  = 0,
        head_queue_size = 3,
        avg_momentum    = None,
        consistency     = None,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        self.lambda_a       = lambda_a
        self.lambda_b       = lambda_b
        self.lambda_idt     = lambda_idt
        self.lambda_consist = lambda_consist
        self.avg_momentum   = avg_momentum
        self.head_config    = head_config or {}
        self.consist_model  = None

        if (lambda_consist > 0) and (consistency is not None):
            self.consist_model \
                = construct_consistency_model(consistency, device)

        assert len(config.data.datasets) == 2, \
            "CycleGAN expects a pair of datasets"


        
        super().__init__(savedir, config, is_train, device)

        self.criterion_gan     = GANLoss(config.loss).to(self.device)
        self.criterion_cycle   = torch.nn.L1Loss()
        self.criterion_idt     = torch.nn.L1Loss()
        self.criterion_consist = torch.nn.L1Loss()
        # self.criterion_consist = self.custom_l1_loss

        self.prev_fake_a = None ## By Grimoire
        self.prev_fake_b = None ## By Grimoire
        self.prev_reco_a = None
        self.lambda_recurrent = 5.0 ## By Grimoire
        self.lambda_depth = 0 ## By Grimoire

        # self.flow_model = ptlflow.get_model('rapidflow_it6', 'things').to(self.device).eval()
        self.flows = [
            torch.from_numpy(np.load('/home/uvcgan2/flow_1_leftdown.npy')).unsqueeze(0).to(device),
            torch.from_numpy(np.load('/home/uvcgan2/flow_2.npy')).unsqueeze(0).to(device),
            torch.from_numpy(np.load('/home/uvcgan2/flow_3_left.npy')).unsqueeze(0).to(device),
            torch.from_numpy(np.load('/home/uvcgan2/flow_4_rightup.npy')).unsqueeze(0).to(device),
            torch.from_numpy(np.load('/home/uvcgan2/flow_5_down.npy')).unsqueeze(0).to(device)
        ]
        # self.flows = np.load('/home/uvcgan2/flow_2.npy')

        # self.flows = torch.from_numpy(self.flows).unsqueeze(0).to(self.device)
        
        encoder='vits'
        # self.depth_model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(self.device).eval()
        # self.depth_transform = Compose([
        #     Resize(
        #         width=320,
        #         height=320,
        #         resize_target=False,
        #         keep_aspect_ratio=True,
        #         ensure_multiple_of=14,
        #         resize_method='lower_bound',
        #         image_interpolation_method=cv2.INTER_CUBIC,
        #     ),
        #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     PrepareForNet(),
        # ])

        
        if self.is_train:
            self.queues = NamedDict(**{
                name : FastQueue(head_queue_size, device = device)
                    for name in [ 'real_a', 'real_b', 'fake_a', 'fake_b' ]
            })

            self.gp = None

            if config.gradient_penalty is not None:
                self.gp = GradientPenalty(**config.gradient_penalty)
    def get_random_flow(self):
        """Randomly selects one of the preloaded flow tensors."""
        return random.choice(self.flows)
        
    def _set_input(self, inputs, domain):
        set_two_domain_input(self.images, inputs, domain, self.device)
        # print(self.images.real_a.shape)
        # print(self.images.real_a)
        
        if self.images.real_a is not None:
            if self.consist_model is not None:
                self.images.consist_real_a \
                    = self.consist_model(self.images.real_a)

        if self.images.real_b is not None:
            if self.consist_model is not None:
                self.images.consist_real_b \
                    = self.consist_model(self.images.real_b)

    def cycle_forward_image(self, real, gen_fwd, gen_bkw):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        fake = gen_fwd(real)
        reco = gen_bkw(fake)

        consist_fake = None

        if self.consist_model is not None:
            consist_fake = self.consist_model(fake)

        return (fake, reco, consist_fake)

    def warp_cycle_forward_image(self, real, gen_fwd): ###########################
        # pylint: disable=no-self-use

        # (N, C, H, W)
        fake = gen_fwd(real)
        random_flow = self.get_random_flow()
        # print(self.flows.shape)
        warped_real_a = warp_flow_torch(real, random_flow)        

        warped_fake_b = gen_fwd(warped_real_a)        
        
        fake_b_warped = warp_flow_torch(fake, random_flow)
        
        # reco = gen_bkw(fake)

        return (warped_fake_b, fake_b_warped)
    

    def idt_forward_image(self, real, gen):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        idt = gen(real)
        return idt

    def forward_dispatch(self, direction):
        if direction == 'ab':
            (
                self.images.fake_b, self.images.reco_a,
                self.images.consist_fake_b
            ) = self.cycle_forward_image(
                self.images.real_a, self.models.gen_ab, self.models.gen_ba
            )

            (
                self.images.warped_fake_b, self.images.fake_b_warped
            ) = self.warp_cycle_forward_image(
                self.images.real_a, self.models.gen_ab
            )

        elif direction == 'ba':
            (
                self.images.fake_a, self.images.reco_b,
                self.images.consist_fake_a
            ) = self.cycle_forward_image(
                self.images.real_b, self.models.gen_ba, self.models.gen_ab
            )

        elif direction == 'aa':
            self.images.idt_a = \
                self.idt_forward_image(self.images.real_a, self.models.gen_ba)

        elif direction == 'bb':
            self.images.idt_b = \
                self.idt_forward_image(self.images.real_b, self.models.gen_ab)

        elif direction == 'avg-ab':
            (
                self.images.fake_b, self.images.reco_a,
                self.images.consist_fake_b
            ) = self.cycle_forward_image(
                self.images.real_a,
                self.models.avg_gen_ab, self.models.avg_gen_ba
            )

        elif direction == 'avg-ba':
            (
                self.images.fake_a, self.images.reco_b,
                self.images.consist_fake_a
            ) = self.cycle_forward_image(
                self.images.real_b,
                self.models.avg_gen_ba, self.models.avg_gen_ab
            )

        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")

    
    
    def forward(self):
        if self.images.real_a is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(direction = 'avg-ab')
                # self._update_previous_results(direction = 'avg-ab')
            else:
                self.forward_dispatch(direction = 'ab')
                self._update_previous_results(direction = 'ab')

        if self.images.real_b is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(direction = 'avg-ba')
                # self._update_previous_results(direction = 'avg-ba')
            else:
                self.forward_dispatch(direction = 'ba')
                # self._update_previous_results(direction = 'ba')
    
    def calculate_difference(self, current, previous):
        return torch.nn.functional.l1_loss(current, previous)
        
    def _update_previous_results(self, direction):
        # 'ba' 방향에서는 이전 결과를 업데이트하지 않음
        if direction != 'ba':
            self.prev_real_a = self.images.real_a.detach().clone() if self.images.real_a is not None else None
            self.prev_real_b = self.images.real_b.detach().clone() if self.images.real_b is not None else None
            self.prev_fake_a = self.images.fake_a.detach().clone() if self.images.fake_a is not None else None
            self.prev_fake_b = self.images.fake_b.detach().clone() if self.images.fake_b is not None else None    
            self.prev_reco_a = self.images.reco_a.detach().clone() if self.images.reco_a is not None else None

    def custom_l1_loss(self, input, target, threshold=0.98):
        """
        input, target: 두 텐서는 같은 차원을 가져야 함
        threshold: 필터링할 픽셀의 밝기 임계값
        """
        # Threshold 기준으로 마스크 생성 (밝기가 threshold 이상인 픽셀만 True)
        mask_input = input > threshold
        mask_target = target > threshold
        combined_mask = mask_input | mask_target
        # save_tensor_as_image(input, '/home/uvcgan2/visual/loss_input.png')
        # save_tensor_as_image(target, '/home/uvcgan2/visual/loss_target.png')
        # save_tensor_as_image(mask_input.float(), '/home/uvcgan2/visual/mask_input.png')
        # save_tensor_as_image(mask_target.float(), '/home/uvcgan2/visual/mask_target.png')
        l1_loss = torch.nn.L1Loss(reduction='mean')
        loss = l1_loss(input[combined_mask], target[combined_mask])
        
        return loss
        
    
    def eval_consist_loss(
        self, consist_real_0, consist_fake_1, lambda_cycle_0
    ):
        return lambda_cycle_0 * self.lambda_consist * self.criterion_consist(
            consist_fake_1, consist_real_0
        ) 
        
    def eval_flow_loss(self, consist_real_0, consist_fake_1, lambda_cycle_0):
            return lambda_cycle_0 * self.criterion_consist(consist_fake_1, consist_real_0) 

    # def eval_consist_loss(
    #     self, consist_real_0, consist_fake_1, lambda_cycle_0
    # ):
    #     return lambda_cycle_0 * self.lambda_consist * self.custom_l1_loss(
    #         consist_fake_1, consist_real_0
    #     )

    def eval_loss_of_cycle_forward(
        self, disc_1, real_0, fake_1, reco_0, fake_queue_1, lambda_cycle_0
    ):
        # pylint: disable=too-many-arguments
        # NOTE: Queue is updated in discriminator backprop
        disc_pred_fake_1 = queued_forward(
            disc_1, fake_1, fake_queue_1, update_queue = False
        )

        loss_gen   = self.criterion_gan(disc_pred_fake_1, True)
        loss_cycle = lambda_cycle_0 * self.criterion_cycle(reco_0, real_0)

        loss = loss_gen + loss_cycle

        return (loss_gen, loss_cycle, loss)

    def eval_loss_of_idt_forward(self, real_0, idt_0, lambda_cycle_0):
        loss_idt = (
              lambda_cycle_0
            * self.lambda_idt
            * self.criterion_idt(idt_0, real_0)
        )

        loss = loss_idt

        return (loss_idt, loss)

    # def recurrent_loss(self, current, previous): ## By Grimoire
    #     # 이전 배치의 결과가 없으면 0 반환
    #     if previous is None:
    #         return 0
    #     # 현재 배치와 이전 배치의 결과 간 L1 손실
    #     return torch.nn.L1Loss()(current, previous)
        
    def recurrent_loss(self, current, previous):
        alpha = 0
        # 이전 배치의 결과가 없으면 0 반환
        if previous is None:
            return 0
        # 현재 배치와 이전 배치의 결과 간 L1 손실
        l1loss = torch.nn.L1Loss()(current, previous)

        # mse_loss = torch.nn.MSELoss()(current, previous)
        # print(current.dtype)
        # print(current.max())
        # print(l1loss)
        ssim_loss = 1 - ms_ssim(current, previous, data_range=1, size_average=True) # return a scalar
        
        mix_loss = alpha * ssim_loss + l1loss
        return mix_loss

    def flow_recurrent_loss(self, current, previous, real_cur, real_pre):
        if previous is None:
            return 0
    
        # Device 설정 (GPU 사용 가정)
        device = self.device  # current 텐서와 같은 디바이스를 사용
    
        # Tensor 형태로 변환 및 디바이스 할당 없이 직접 사용 (이미 텐서로 가정)
        real_cur = real_cur.float().to(device)  # float 형태로 보장
        real_pre = real_pre.float().to(device)
        save_tensor_as_image(current, '/home/uvcgan2/visual/current_1.png')

        current = current.float().to(device)
        previous = previous.float().to(device)
    
        # 모델 및 입력 준비
        io_adapter = IOAdapter(self.flow_model, real_pre.shape[2:])  # shape[2:]로 높이와 너비 추출
        prepared_inputs = prepare_tensor_inputs(io_adapter, images=[real_pre, real_cur], device=device)
        predictions = self.flow_model(prepared_inputs)
        flow = predictions['flows'][0, 0].detach()  # 예측된 flow 가져오기 torch.Size([2, 320, 320])
        flow = flow.unsqueeze(0)  # Adds a batch dimension -> (1, 2, 320, 320)
        # save_tensor_as_image(current, '/home/uvcgan2/visual/current_2.png')
        # warp_flow 함수 호출 (Tensor 연산으로 구현)
        warped_image_tensor = warp_flow_torch(previous, flow).detach()
        # save_tensor_as_image(warped_image_tensor, '/home/uvcgan2/visual/warped_image.png')

        mask_current = current > 0.98
        mask_warped_image_tensor = warped_image_tensor > 0.98
        combined_mask = mask_current | mask_warped_image_tensor

        # L1 Loss 계산
        l1loss = torch.nn.L1Loss()(current[combined_mask], warped_image_tensor[combined_mask])
        # loss = l1_loss(current[combined_mask], warped_image_tensor[combined_mask])
        # print('l1loss is: ',l1loss)
        return l1loss

        
    # def calculate_depth_loss(self, real_images, fake_images):
    # # 배치 차원 제거: [1, 3, 320, 320] -> [3, 320, 320]
    #     real_images = real_images.squeeze(0).detach().numpy()
    #     fake_images = fake_images.squeeze(0).detach().numpy()
        
    #     real_images = np.transpose(real_images, (1, 2, 0))
    #     fake_images = np.transpose(fake_images, (1, 2, 0))
        
    #     # 입력 이미지에 대한 변환 적용
    #     real_images_transformed = self.depth_transform({'image': real_images})['image']
    #     fake_images_transformed = self.depth_transform({'image': fake_images})['image']

    #     # 변환된 이미지 저장
        
    #     real_images_transformed = torch.from_numpy(real_images_transformed).unsqueeze(0).to(self.device)
    #     fake_images_transformed = torch.from_numpy(fake_images_transformed).unsqueeze(0).to(self.device)
        
    #     with torch.no_grad():
    #         # Depth 추정을 위해 배치 차원 추가: [3, 320, 320] -> [1, 3, 320, 320]
    #         real_depth = self.depth_model(real_images_transformed)
    #         fake_depth = self.depth_model(fake_images_transformed)
    #     # Depth loss 계산 (예: L1 loss)
    #     # depth_loss = torch.nn.functional.l1_loss(real_depth, fake_depth)
    #     depth_loss = torch.nn.MSEloss()(real_depth, fake_depth)
    #     return depth_loss
    
    def backward_gen(self, direction):
        difference_threshold = 0.3

        if direction == 'ab':
            (self.losses.gen_ab, self.losses.cycle_a, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc_b,
                    self.images.real_a, self.images.fake_b, self.images.reco_a,
                    self.queues.fake_b, self.lambda_a
                )

            if self.consist_model is not None:
                self.losses.consist_a = self.eval_consist_loss(
                    self.images.consist_real_a, self.images.consist_fake_b,
                    self.lambda_a
                )
                # print('consist_loss_a: ', self.losses.consist_a)
                loss += self.losses.consist_a

            self.losses.flow = self.eval_flow_loss(self.images.fake_b_warped, self.images.warped_fake_b, 10)
            loss += self.losses.flow
            
            activate_recurrent_loss = False
            if self.prev_fake_b is not None:
                # save_tensor_as_image(self.images.real_a, '/home/uvcgan2/visual/backward_gen_curr_real_a.png')
                # save_tensor_as_image(self.prev_real_a, '/home/uvcgan2/visual/backward_gen_prev_real_a.png')
                difference = self.calculate_difference(self.images.real_a, self.prev_real_a)
                # print('pixel_difference :', difference)
                if difference > difference_threshold:
                    activate_recurrent_loss = False
                    
            if activate_recurrent_loss and self.prev_fake_b is not None:
                recurrent_loss = self.flow_recurrent_loss(self.images.fake_b, self.prev_fake_b, self.images.real_a, self.prev_real_a)
                # print('Recurrent_loss: ', recurrent_loss)
                recurrent_loss_reco = self.flow_recurrent_loss(self.images.reco_a, self.prev_reco_a, self.images.real_a, self.prev_real_a)
                # print('Recurrent_loss_reco: ', recurrent_loss_reco)
                loss += self.lambda_recurrent * recurrent_loss + self.lambda_recurrent * recurrent_loss

            # if self.lambda_depth > 0:
            #     depth_loss = self.calculate_depth_loss(self.images.real_a, self.images.reco_a)
            #     loss += self.lambda_depth * depth_loss

        elif direction == 'ba':
            (self.losses.gen_ba, self.losses.cycle_b, loss) \
                = self.eval_loss_of_cycle_forward(
                    self.models.disc_a,
                    self.images.real_b, self.images.fake_a, self.images.reco_b,
                    self.queues.fake_a, self.lambda_b
                )

            if self.consist_model is not None:
                self.losses.consist_b = self.eval_consist_loss(
                    self.images.consist_real_b, self.images.consist_fake_a,
                    self.lambda_b
                )
                print('consist_loss_b: ', self.losses.consist_b)
                loss += self.losses.consist_b

        elif direction == 'aa':
            (self.losses.idt_a, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_a, self.images.idt_a, self.lambda_a
                )

        elif direction == 'bb':
            (self.losses.idt_b, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_b, self.images.idt_b, self.lambda_b
                )
        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")

        # 이전 결과 업데이트
        # print('total loss: ', loss)
        loss.backward()

        self._update_previous_results(direction)
        
    def backward_discriminator_base(
        self, model, real, fake, queue_real, queue_fake
    ):
        # pylint: disable=too-many-arguments
        loss_gp = None

        if self.gp is not None:
            loss_gp = self.gp(
                model, fake, real,
                model_kwargs_fake = { 'extra_bodies' : queue_fake.query() },
                model_kwargs_real = { 'extra_bodies' : queue_real.query() },
            )
            loss_gp.backward()

        pred_real = queued_forward(
            model, real, queue_real, update_queue = True
        )
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake = queued_forward(
            model, fake, queue_fake, update_queue = True
        )
        loss_fake = self.criterion_gan(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5
        loss.backward()

        return (loss_gp, loss)

    def backward_discriminators(self):
        fake_a = self.images.fake_a.detach()
        fake_b = self.images.fake_b.detach()

        loss_gp_b, self.losses.disc_b \
            = self.backward_discriminator_base(
                self.models.disc_b, self.images.real_b, fake_b,
                self.queues.real_b, self.queues.fake_b
            )

        if loss_gp_b is not None:
            self.losses.gp_b = loss_gp_b

        loss_gp_a, self.losses.disc_a = \
            self.backward_discriminator_base(
                self.models.disc_a, self.images.real_a, fake_a,
                self.queues.real_a, self.queues.fake_a
            )

        if loss_gp_a is not None:
            self.losses.gp_a = loss_gp_a

    def optimization_step_gen(self):
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], False)
        self.optimizers.gen.zero_grad(set_to_none = True)

        dir_list = [ 'ab', 'ba' ]
        if self.lambda_idt > 0:
            dir_list += [ 'aa', 'bb' ]

        for direction in dir_list:
            self.forward_dispatch(direction)
            self.backward_gen(direction)

        self.optimizers.gen.step()

    def optimization_step_disc(self):
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], True)
        self.optimizers.disc.zero_grad(set_to_none = True)

        self.backward_discriminators()

        self.optimizers.disc.step()

    def _accumulate_averages(self):
        update_average_model(
            self.models.avg_gen_ab, self.models.gen_ab, self.avg_momentum
        )
        update_average_model(
            self.models.avg_gen_ba, self.models.gen_ba, self.avg_momentum
        )

    def optimization_step(self):
        self.optimization_step_gen()
        self.optimization_step_disc()

        if self.avg_momentum is not None:
            self._accumulate_averages()

