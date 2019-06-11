from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math
from utils import resize_image

#################################################################################
# You will need to fill in the missing code in this file
#################################################################################


#################################################################################
# Part I: Understanding Convolutions
#################################################################################
class CustomConv2DFunction(Function):

  @staticmethod
  def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
    """
    Forward propagation of convolution operation.
    We only consider square filters with equal stride/padding in width and height!

    Args:
      input_feats: input feature map of size N * C_i * H * W
      weight: filter weight of size C_o * C_i * K * K
      bias: (optional) filter bias of size C_o
      stride: (int, optional) stride for the convolution. Default: 1
      padding: (int, optional) Zero-padding added to both sides of the input. Default: 0

    Outputs:
      output: responses of the convolution  w*x+b

    """
    # sanity check
    assert weight.size(2) == weight.size(3)
    assert input_feats.size(1) == weight.size(1)
    assert isinstance(stride, int) and (stride > 0)
    assert isinstance(padding, int) and (padding >= 0)

    # save the conv params
    kernel_size = weight.size(2)
    ctx.batch_size = input_feats.size(0)
    ctx.stride = stride
    ctx.padding = padding
    ctx.input_height = input_feats.size(2)
    ctx.input_width = input_feats.size(3)

    # make sure this is a valid convolution
    assert kernel_size <= (input_feats.size(2) + 2 * padding)
    assert kernel_size <= (input_feats.size(3) + 2 * padding)

    #################################################################################
    # Fill in the code here
    #################################################################################
    C_o, C_i = weight.size()[:2]
    H_o = 1 + int((ctx.input_height + 2*padding - kernel_size) // stride)
    W_o = 1 + int((ctx.input_width + 2*padding - kernel_size) // stride)
    windows = unfold(input_feats, kernel_size, padding=padding, stride=stride)
    kernels = weight.view(C_o, C_i * kernel_size * kernel_size)
    output = torch.einsum('ncw,oc->now', windows, kernels).view(-1, C_o, H_o, W_o) \
            + (bias.view(1,-1,1,1).expand(ctx.batch_size,-1,H_o,W_o))

    # save for backward (you need to save the unfolded tensor into ctx)
    ctx.save_for_backward(windows, weight, bias)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    """
    Backward propagation of convolution operation

    Args:
      grad_output: gradients of the outputs

    Outputs:
      grad_input: gradients of the input features
      grad_weight: gradients of the convolution weight
      grad_bias: gradients of the bias term

    """
    # unpack tensors and initialize the grads
    windows, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None

    # recover the conv params
    kernel_size = weight.size(2)
    stride = ctx.stride
    padding = ctx.padding
    batch_size = ctx.batch_size
    input_height = ctx.input_height
    input_width = ctx.input_width

    #################################################################################
    # Fill in the code here
    #################################################################################
    # compute the gradients w.r.t. input and params
    C_i = weight.size(1)
    C_o, H_o, W_o = grad_output.size()[1:]
    if ctx.needs_input_grad[0]:
        grad_windows = torch.einsum('nouv,oiab->niabuv', grad_output, weight)\
                            .view(batch_size, C_i*kernel_size*kernel_size, -1)
        grad_input = fold(grad_windows, (input_height, input_width),
                        kernel_size, padding=padding, stride=stride)
    if ctx.needs_input_grad[1]:
        grad_weight = torch.einsum('nouv,ncuv->oc',
                                grad_output,
                                windows.view(batch_size,-1,H_o,W_o))\
                            .view(C_o,C_i,kernel_size,kernel_size)

    if bias is not None and ctx.needs_input_grad[2]:
      # compute the gradients w.r.t. bias (if any)
      grad_bias = grad_output.sum((0,2,3))

    return grad_input, grad_weight, grad_bias, None, None

custom_conv2d = CustomConv2DFunction.apply

class CustomConv2d(Module):
  """
  The same interface as torch.nn.Conv2D
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
         padding=0, dilation=1, groups=1, bias=True):
    super(CustomConv2d, self).__init__()
    assert isinstance(kernel_size, int), "We only support squared filters"
    assert isinstance(stride, int), "We only support equal stride"
    assert isinstance(padding, int), "We only support equal padding"
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    # not used (for compatibility)
    self.dilation = dilation
    self.groups = groups

    # register weight and bias as parameters
    self.weight = nn.Parameter(torch.Tensor(
      out_channels, in_channels, kernel_size, kernel_size))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_channels))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
  	# initialization using Kaiming uniform
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in)
      nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, input):
    # call our custom conv2d op
    return custom_conv2d(input, self.weight, self.bias, self.stride, self.padding)

  def extra_repr(self):
    s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
       ', stride={stride}, padding={padding}')
    if self.bias is None:
      s += ', bias=False'
    return s.format(**self.__dict__)

#################################################################################
# Part II: Design and train a network
#################################################################################
class SimpleNet(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(SimpleNet, self).__init__()
    # you can start from here and create a better model
    self.features = nn.Sequential(
      # conv1 block: 3x conv 3x3
      conv_op(3, 64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv2 block: simple bottleneck
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv3 block: simple bottleneck
      conv_op(256, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv4 block: conv 3x3
      conv_op(256, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
    )
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    # you can implement adversarial training here
    if self.training:
      # generate adversarial sample based on x
      pgd = PGDAttack(nn.CrossEntropyLoss())
      self.training = False
      x = pgd.perturb(self, x)
      self.training = True
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

class SimpleNetBN2D(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(SimpleNet, self).__init__()
    # you can start from here and create a better model
    self.features = nn.Sequential(
      # batch-norm the input, and each pool output
      nn.BatchNorm2d(3),
      # conv1 block: 3x conv 3x3
      conv_op(3, 64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      # conv2 block: simple bottleneck
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(256),
      # conv3 block: simple bottleneck
      conv_op(256, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(256),
      # conv4 block: conv 3x3
      conv_op(256, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(512),
    )
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    # you can implement adversarial training here
    if self.training:
      # generate adversarial sample based on x
      pgd = PGDAttack(nn.CrossEntropyLoss())
      self.training = False
      x = pgd.perturb(self, x)
      self.training = True
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

# Depthwise convolution as used by MobileNets
def conv_dw(inp, out, kernel_size=3, stride=1, padding=1, dilation=1,
            bias=False):
  return nn.Sequential(
           nn.Conv2d(inp, inp, kernel_size, 
                     stride=stride, padding=padding, dilation=dilation,
                     groups=inp, bias=bias),
           nn.Conv2d(inp, out, 1, stride=1, padding=0, bias=bias),
           )

# Stealing from torchvision's ResNet implementation, but using depthwise
# convolution
def conv3x3(inp, out, stride=1):
  return conv_dw(inp, out, kernel_size=3, stride=stride, padding=1,
                bias=False)

def conv1x1(inp, out, stride=1):
  return conv_dw(inp, out, kernel_size=1, padding=0, 
                stride=stride, bias=False)

def downsample(inp, out, stride):
  return nn.Sequential(
            conv1x1(inp, out, stride=stride),
            nn.BatchNorm2d(out),
        )

class BasicBlock(nn.Module):
  # Fixed stride = 1!
  def __init__(self, inp, out, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inp, out, stride=stride)
    self.bn1 = nn.BatchNorm2d(out)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(out, out)
    self.bn2 = nn.BatchNorm2d(out)
    self.downsample = downsample

  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity 
    out = self.relu(out)
    return out

class Bottleneck(nn.Module):
  def __init__(self, inp, out, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = conv1x1(inp, out)
    self.bn1 = nn.BatchNorm2d(out)
    self.conv2 = conv3x3(out, out, stride=stride)
    self.bn2 = nn.BatchNorm2d(out)
    self.conv3 = conv1x1(out, inp)
    self.bn3 = nn.BatchNorm2d(inp)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample

  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)
    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)
    return out


class LessSimpleNetBN2D_ConvDW(nn.Module):
  # a simple CNN for image classifcation, with batch norm everywhere
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(LessSimpleNetBN2D_ConvDW, self).__init__()

    # you can start from here and create a better model
    self.features = nn.Sequential(
      nn.BatchNorm2d(3),
      # conv1 block: 3x conv 3x3
      conv_dw(3, 64, kernel_size=7, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      BasicBlock(64, 256, stride=2, downsample=downsample(64,256,2)),
      BasicBlock(256, 512, downsample=downsample(256,512,1)),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv2 block: simple bottleneck
      nn.BatchNorm2d(512),
      Bottleneck(512, 128, stride=2, downsample=downsample(512,512,2)),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv3 block: simple bottleneck
      nn.BatchNorm2d(512),
      Bottleneck(512, 128, stride=2, downsample=downsample(512,512,2)),
    )
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

class SimpleNetBN2D_ConvDW(nn.Module):
  # a simple CNN for image classifcation, with batch norm everywhere
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(SimpleNetBN2D_ConvDW, self).__init__()
    
    def conv_dw(inp, out, kernel_size=3, stride=1, padding=1, dilation=1):
        return nn.Sequential(
               nn.Conv2d(inp, inp, kernel_size, 
                         stride=stride, padding=padding, dilation=dilation,
                         groups=inp, bias=True),
               nn.Conv2d(inp, out, 1, stride=1, padding=0, bias=True),
               )

    # you can start from here and create a better model
    self.features = nn.Sequential(
      nn.BatchNorm2d(3),
      # conv1 block: 3x conv 3x3
      conv_dw(3, 64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv2 block: simple bottleneck
      nn.BatchNorm2d(64),
      conv_dw(64, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_dw(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_dw(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv3 block: simple bottleneck
      nn.BatchNorm2d(256),
      conv_dw(256, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_dw(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_dw(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv4 block: conv 3x3
      nn.BatchNorm2d(256),
      conv_dw(256, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(512),
    )
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

class VGGNet(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(NotSoSimpleNet, self).__init__()
    self.features = nn.Sequential(
      #torch.nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      nn.BatchNorm2d(3),
      # conv1_1 block: 3x conv 3x3
      conv_op(3, 64, kernel_size=3,  padding=1),
      # Relu 1_1
      nn.ReLU(inplace=True),
      # conv 1_2 block
      conv_op(64, 64 ,kernel_size=3, padding=1),
      # Relu 1_2
      nn.ReLU(inplace=True),

      #pool 1
      nn.MaxPool2d(kernel_size=2,stride=2),
      nn.BatchNorm2d(64),
      #conv 2_1 block
      conv_op(64,128 , kernel_size=3, padding=1),
      # Relu 2_1
      nn.ReLU(inplace=True),

      #conv 2_2 block
      conv_op(128,128 , kernel_size=3 , padding=1),
      # Relu 2_2
      nn.ReLU(inplace=True),

      #pool 2
      nn.MaxPool2d(kernel_size=2 , stride=2),
      nn.BatchNorm2d(128),
      # conv 3_1 block
      conv_op(128 , 256 , kernel_size=3 , padding=1),
      # Relu 3_1
      nn.ReLU(inplace= True),

      #conv 3_2
      conv_op(256, 256 , kernel_size=3, padding=1),
      #ReLu 3_2
      nn.ReLU(inplace=True),
      # conv 3_3 block
      conv_op(256 ,256 ,kernel_size=3, padding=1),
      # Relu 3_3 block
      nn.ReLU(inplace=True),
      # pool 3
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.BatchNorm2d(256),
      #conv 4_1
      conv_op(256 , 512 , kernel_size= 3 , padding=1),
      # Relu 4_1
      nn.ReLU(inplace=True),
      # conv 4_2
      conv_op(512 , 512 , kernel_size= 3, padding=1),
      # ReLu 4_2
      nn.ReLU(inplace= True),
      # conv 4_3
      conv_op(512 ,512 , kernel_size=3 , padding=1 ),
      # Relu 4_3
      nn.ReLU(inplace=True),
      # pool 4
      nn.MaxPool2d(kernel_size=2 ,stride=2),
      nn.BatchNorm2d(512),
      # conv 5_1
      conv_op(512, 512 , kernel_size=3 ,padding=1),
      #relu 5_1
      nn.ReLU(inplace=True),
      #conv 5_2
      conv_op(512, 512 , kernel_size=3 , padding=1),
      # relu 5_2
      nn.ReLU(inplace=True),
      #conv 5_3
      conv_op(512, 512 , kernel_size=3, padding=1),
      #relu
      nn.ReLU(inplace=True),
      #pool 5
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.BatchNorm2d(512),

    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc=nn.Linear(512,num_classes)

  def forward(self, x):
    # you can implement adversarial training here
    # if self.training:
    #   # generate adversarial sample based on x
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

class MobileNet(nn.Module):

    def __init__(self,conv_op=nn.Conv2d, num_classes=100):
        super(MobileNet,self).__init__()

        def conv_dw(inp, out , stride): 
            return nn.Sequential( 
               nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
               nn.BatchNorm2d(inp),
               nn.ReLU(inplace=True),
               nn.Conv2d(inp, out, 1, 1, 0, bias=False),
               nn.BatchNorm2d(out),
               nn.ReLU(inplace=True),   
               )

        def conv_batch_norm(inp, out , stride):
            return nn.Sequential(
                    nn.Conv2d(inp, out, 3, stride, 1 , bias=False),
                    nn.BatchNorm2d(out), 
                    nn.ReLU(inplace=True),
                    )
        self.features= nn.Sequential(
                conv_batch_norm(3,32,2),
                conv_dw(32,64,1),
                conv_dw(64,128,2),
                conv_dw(128,128,1),
                conv_dw(128,256,2),
                conv_dw(256,256,1),
                conv_dw(256,512,2),
                conv_dw(512,512,1),
                conv_dw(512,512,1),
                conv_dw(512,512,1),
                conv_dw(512,512,1),
                conv_dw(512,512,1),
                conv_dw(512,512,2),
                conv_dw(512,512,1),
                nn.AvgPool2d(7),
                )
        self.fc=nn.Linear(512,100)
        
    def forward(self,x):
         x=self.features(x)
         x=x.view(x.size(0),-1)
         x=self.fc(x)
         return x 

# change this to your model!
#default_model = SimpleNet
#default_model = SimpleNetBN2D_ConvDW
default_model = MobileNet 
#################################################################################
# Part III: Adversarial samples and Attention
#################################################################################
class PGDAttack(object):
  def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.1):
    """
    Attack a network by Project Gradient Descent. The attacker performs
    k steps of gradient descent of step size a, while always staying
    within the range of epsilon from the input image.

    Args:
      loss_fn: loss function used for the attack
      num_steps: (int) number of steps for PGD
      step_size: (float) step size of PGD
      epsilon: (float) the range of acceptable samples
               for our normalization, 0.1 ~ 6 pixel levels
    """
    self.loss_fn = loss_fn
    self.num_steps = num_steps
    self.step_size = step_size
    self.epsilon = epsilon

  def perturb(self, model, input):
    """
    Given input image X (torch tensor), return an adversarial sample
    (torch tensor) using PGD of the least confident label.

    See https://openreview.net/pdf?id=rJzIBfZAb

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) an adversarial sample of the given network
    """
    # clone the input tensor and disable the gradients
    output = input.clone()
    input.requires_grad = False

    # loop over the number of steps
    for _ in range(self.num_steps):
      output.requires_grad = True
      pred = model(output)
      loss = self.loss_fn(pred, torch.argmin(pred, dim=1))
      loss.backward()
      output = output - self.step_size * torch.sign(output.grad)
      output.detach_()
    delta = torch.clamp(output - input, -self.epsilon, self.epsilon)
    output = input + delta
    return output

default_attack = PGDAttack


class GradAttention(object):
  def __init__(self, loss_fn):
    """
    Visualize a network's decision using gradients

    Args:
      loss_fn: loss function used for the attack
    """
    self.loss_fn = loss_fn

  def explain(self, model, input):
    """
    Given input image X (torch tensor), return a saliency map
    (torch tensor) by computing the max of abs values of the gradients
    given by the predicted label

    See https://arxiv.org/pdf/1312.6034.pdf

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) a saliency map of size N * 1 * H * W
    """
    # make sure input receive grads
    input.requires_grad = True
    if input.grad is not None:
      input.grad.zero_()

    #################################################################################
    # Fill in the code here
    #################################################################################
    pred = model(input)
    loss = self.loss_fn(pred, torch.argmax(pred, dim=1))
    loss.backward()
    output = torch.max(torch.abs(input.grad), 1)[0]\
                    .view(input.size(0), 1, input.size(2), input.size(3))
    return output

default_attention = GradAttention

def vis_grad_attention(input, vis_alpha=2.0, n_rows=10, vis_output=None):
  """
  Given input image X (torch tensor) and a saliency map
  (torch tensor), compose the visualziations

  Args:
    input: (torch tensor) input image of size N * C * H * W
    output: (torch tensor) input map of size N * 1 * H * W

  Outputs:
    output: (torch tensor) visualizations of size 3 * HH * WW
  """
  # concat all images into a big picture
  input_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
  if vis_output is not None:
    output_maps = make_grid(vis_output.cpu(), nrow=n_rows, normalize=True)

    # somewhat awkward in PyTorch
    # add attention to R channel
    mask = torch.zeros_like(output_maps[0, :, :]) + 0.5
    mask = (output_maps[0, :, :] > vis_alpha * output_maps[0,:,:].mean())
    mask = mask.float()
    input_imgs[0,:,:] = torch.max(input_imgs[0,:,:], mask)
  output = input_imgs
  return output

default_visfunction = vis_grad_attention
