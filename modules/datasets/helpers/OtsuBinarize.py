import torch
import numpy as np
import cv2
from PIL import Image


class OtsuBinarize:
    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.dim() != 3 or input_tensor.shape[0] != 1:
            raise ValueError("Expected single-channel (1, H, W) tensor.")

        img_np = (input_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        # Perform Otsu binarization
        _, binary_img_np = cv2.threshold(
            img_np, 25, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU
        )

        binary_tensor = torch.from_numpy(binary_img_np).float().unsqueeze(0) / 255.0

        # Will return a torch.int8 tensor
        return binary_tensor
