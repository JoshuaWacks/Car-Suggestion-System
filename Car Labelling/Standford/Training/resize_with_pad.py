import torchvision.transforms.functional as F

class ResizeWithPad:
    def __init__(self, target_size, fill_value=0, padding_mode='constant'):
        self.target_size = target_size  # (height, width)
        self.fill_value = fill_value
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Get original image dimensions
        img_w, img_h = img.size
        target_h, target_w = self.target_size

        # Calculate scaling factor to fit within target_size while maintaining aspect ratio
        scale = min(target_w / img_w, target_h / img_h)

        # Calculate new dimensions after scaling
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        # Resize the image
        img = F.resize(img, (new_h, new_w))

        # Calculate padding
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)

        # Apply padding
        img = F.pad(img, padding, fill=self.fill_value, padding_mode=self.padding_mode)

        return img