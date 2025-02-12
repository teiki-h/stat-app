#############
## DEFINING HOOKS TO TRACE AND LOCATE FACTUAL ASSOCIATIONS
#############

import torch

def output_hook(module, input, output):
    """
    Forward hook to print the output of a Pytorch module during forward pass. Also prints output shape.

    Args:
        module (torch.nn.Module): The module to which this hook is attached.
        output (tuple of torch.Tensor): The output tensor(s) produced by the module.
    """

    print(f'{module} : output')
    print(output.shape)
    print(output)


def input_hook(module, input, output):
    """
    Forward hook to print the input of a Pytorch module during forward pass

    Args:
        module (torch.nn.Module): The module to which this hook is attached.
        input (tuple of torch.Tensor): The input tensors passed to the module.
        output (tuple of torch.Tensor): The output tensor(s) produced by the module.
    """

    print(f'{module} : input')
    print(input)


def naive_noise_hook(module, input, output):
    """
    Forward hook adding some random noise to a PyTorch module output

    Args:
        module (torch.nn.Module): The module to which this hook is attached.
        input (tuple of torch.Tensor): The input tensors passed to the module.
        output (tuple of torch.Tensor): The output tensor(s) produced by the module.
    """

    noise = torch.randn_like(output)
    return output+noise


def noise_hook(module,input,output):
    """
    Forward hook adding some random noise to a PyTorch module output

    Args:
        module (torch.nn.Module): The module to which this hook is attached.
        input (tuple of torch.Tensor): The input tensors passed to the module.
        output (tuple of torch.Tensor): The output tensor(s) produced by the module.
    """
    
    noise = torch.randn_like(output)#*sqrt(3*variance)
    noisy_output = output + noise * masks_tensor.unsqueeze(-1).float()
    return noisy_output
#### TO-DO: AJUSTER LE BRUIT À 3 FOIS LA VARIANCE