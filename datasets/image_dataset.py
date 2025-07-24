from torchvision.datasets import ImageFolder
from torchvision import transforms
from datasets import register
from einops import repeat
from PIL import Image
import numpy as np
         

class CustomCenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_image):
        # Your center_crop_arr logic here, assuming img is a PIL image
        while min(*pil_image.size) >= 2 * self.size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - self.size) // 2
        crop_x = (arr.shape[1] - self.size) // 2
        return Image.fromarray(arr[crop_y: crop_y + self.size, crop_x: crop_x + self.size])
   

@register('image_dataset')
class ImgDataSet(ImageFolder):
    def __init__(self, root_path, crop_size, cls_vid_num):
        transform = transforms.Compose([
            # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            CustomCenterCrop(crop_size),
            # transforms.Resize(crop_size),
            # transforms.CenterCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])        
        super().__init__(root_path, transform=transform)

        # create a subset 
        subset_size = int(float(cls_vid_num))
        if 0 < subset_size < len(self.samples):
            self.samples, self.targets = self.samples[:subset_size], self.targets[:subset_size]
            # indices = np.arange(subset_size)
            # self.samples = [self.samples[i] for i in indices]
            # self.targets = [self.targets[i] for i in indices]

    def __getitem__(self, idx):
        img, y = super().__getitem__(idx)
        return {'video': img, 'label': y}

        # vid = repeat(img, 'c h w -> c t h w', t=self.frame_num)

# Deprecated
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
