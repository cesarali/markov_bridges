import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from markov_bridges.configs.config_classes.data.graphs_configs import GraphDataloaderGeometricConfig

SqueezeTransform = transforms.Lambda(lambda x: x.squeeze())
FlattenTransform = transforms.Lambda(lambda x: x.reshape(x.shape[0], -1))
UnFlattenTransform = transforms.Lambda(lambda x: x.reshape(x.shape[0],
                                                           int(np.sqrt(x.shape[1])),
                                                           int(np.sqrt(x.shape[1]))))

BinaryTensorToSpinsTransform = transforms.Lambda(lambda binary_tensor: (-1.) ** (binary_tensor + 1))


# Create a custom transformation class
class CorrectEMNISTOrientation(object):
    def __call__(self, img):
        return transforms.functional.rotate(img, -90).transpose(Image.FLIP_LEFT_RIGHT)

class ToUpperDiagonalIndicesTransform:

    def __call__(self, tensor):
        if  len(tensor.shape) == 3:
            batch_size = tensor.shape[0]
            # Get the upper diagonal entries without zero-padding with the batch as the first dimension
            upper_diagonal_entries = tensor.masked_select(torch.triu(torch.ones_like(tensor), diagonal=1).bool())
            upper_diagonal_entries = upper_diagonal_entries.reshape(batch_size, -1)
            return upper_diagonal_entries
        else:
            raise Exception("Wrong Tensor Shape in Transform")

class FromUpperDiagonalTransform:

    def __call__(self, upper_diagonal_tensor):
        assert len(upper_diagonal_tensor.shape) == 2
        number_of_upper_entries = upper_diagonal_tensor.shape[1]
        batch_size = upper_diagonal_tensor.shape[0]

        matrix_size = int(.5 * (1 + np.sqrt(1 + 8 * number_of_upper_entries)))

        # Create a zero-filled tensor to hold the full matrices
        full_matrices = torch.zeros(batch_size, matrix_size, matrix_size, device=upper_diagonal_tensor.device)

        # Get the indices for the upper diagonal part of the matrices
        upper_tri_indices = torch.triu_indices(matrix_size, matrix_size, offset=1, device=upper_diagonal_tensor.device)

        # Fill the upper diagonal part of the matrices
        full_matrices[:, upper_tri_indices[0], upper_tri_indices[1]] = upper_diagonal_tensor

        # Transpose and fill the lower diagonal part to make the matrices symmetric
        full_matrices = full_matrices + full_matrices.transpose(1, 2)

        return full_matrices

class UnsqueezeTensorTransform:

    def __init__(self,axis=0):
        self.axis = axis
    def __call__(self, tensor:torch.Tensor):
        return tensor.unsqueeze(self.axis)

class BinaryTensorToSpinsTransform:

    def __call__(self,binary_tensor):
        spins = (-1.) ** (binary_tensor + 1)
        return spins

class SpinsToBinaryTensor:
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be transformed.
        Returns:
            Tensor: Transformed tensor.
        """
        # Create a copy of the input tensor to avoid modifying the original tensor
        transformed_tensor = tensor.clone()
        # Replace -1. with 0. in the tensor
        transformed_tensor[transformed_tensor == -1.] = 0.

        return transformed_tensor

def get_transforms(config:GraphDataloaderGeometricConfig):
    """
    :param config:

    :return: transform_list,inverse_transform_list
    """
    if config.full_adjacency:
        transform_list = [FlattenTransform]
        inverse_transform_list = [UnFlattenTransform]
    else:
        transform_list = [ToUpperDiagonalIndicesTransform()]
        inverse_transform_list = [FromUpperDiagonalTransform()]
    
    #if config.flatten:
    #else:
    #    if config.full_adjacency:
    #        transform_list = []
    #        inverse_transform_list = []
    #    else:  # no flatten no full adjacency
    #       raise Exception("No Flatten and No Full Adjacency incompatible for data")
        
    transform_list = transforms.Compose(transform_list)
    inverse_transform_list = transforms.Compose(inverse_transform_list)

    return transform_list,inverse_transform_list

def get_expected_shape(max_node_num,flatten_adjacency,full_adjacency):
    """
    obtains the expected shape of the dataset
    """
    number_of_upper_entries = int(max_node_num*(max_node_num-1.)*.5)
    if flatten_adjacency:
        if full_adjacency:
            D = max_node_num * max_node_num
            shape = [D]
        else:
            D = number_of_upper_entries
            shape = [D]
    else:
        if full_adjacency:
            D = max_node_num * max_node_num
            shape = [max_node_num, max_node_num]
        else:
            raise ValueError("No Flatten and No Full Adjacency incompatible for data")
    return D, shape