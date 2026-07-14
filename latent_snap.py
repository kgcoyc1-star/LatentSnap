import torch
import torch.nn.functional as F

class LatentSnap:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "divisor": (["16", "32", "64"], {"default": "64"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("latent", "image", "width", "height")
    FUNCTION = "run"
    CATEGORY = "my_nodes/latent"

    def round_to_multiple(self, value, multiple):
        rounded = round(value / multiple) * multiple
        return max(multiple, rounded)

    def run(self, divisor, batch_size, latent=None, image=None, width=512, height=512):
        m = int(divisor)

        if latent is not None:
            samples = latent["samples"]
            src_h = samples.shape[2] * 8
            src_w = samples.shape[3] * 8
        elif image is not None:
            src_h = image.shape[1]
            src_w = image.shape[2]
        else:
            src_w = width
            src_h = height

        new_w = self.round_to_multiple(src_w, m)
        new_h = self.round_to_multiple(src_h, m)

        latent_h = new_h // 8
        latent_w = new_w // 8
        empty_latent = torch.zeros([batch_size, 4, latent_h, latent_w])

        if image is not None:
            img = image.permute(0, 3, 1, 2)
            img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
            out_image = img.permute(0, 2, 3, 1)
        else:
            out_image = torch.zeros([batch_size, new_h, new_w, 3])

        print(f"[LatentSnap] {src_w}x{src_h} -> {new_w}x{new_h} (кратно {m})")

        return ({"samples": empty_latent}, out_image, new_w, new_h)


NODE_CLASS_MAPPINGS = {
    "LatentSnap": LatentSnap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSnap": "Latent Snap (÷16/32/64)"
}