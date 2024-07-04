import torch

__all__ = [
    "spatial_reduce_crop",
    "SpatialReduceCrop"
]
def spatial_reduce_crop(x:torch.Tensor, crop:int = 1):
    """Crops spatial dim sizes in a BC* tensor by `num`.
    For example, if `num = 1`, (16, 3, 129, 129) -> (16, 3, 128, 128).
    This crops at the end. Useful to crop padding which can only add even size."""
    slices = [slice(None, -crop) for _ in range(x.ndim - 2)]
    return x[:,:,*slices]

class SpatialReduceCrop(torch.nn.Module):
    def __init__(self, crop:int = 1):
        super().__init__()
        self.crop = crop

    def forward(self, x:torch.Tensor):
        return spatial_reduce_crop(x, crop=self.crop)