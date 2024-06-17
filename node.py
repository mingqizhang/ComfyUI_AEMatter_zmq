import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import AEMatterModel

device = "cuda" if torch.cuda.is_available() else "cpu"
folder_paths.folder_names_and_paths["AEMatter"] = ([os.path.join(folder_paths.models_dir, "AEMatter")], folder_paths.supported_pt_extensions)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def tensor2numpy(image):
    return np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class AEMatter_ModelLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("AEMatter"),),
            }
        }

    RETURN_NAMES = ("AEMatterModel",)
    FUNCTION = "load_model"
    CATEGORY = "AEMatter"
    RETURN_TYPES = ("AEMMODEL",)

    def load_model(self, model):
        net = AEMatterModel.AEMatter()
        model_path = folder_paths.get_full_path("AEMatter", model)
        net.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
        net.to(device)
        net.eval()
        return [net]


class AEMatter_Apply:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "AEMatterModel": ("AEMMODEL",),
                "image": ("IMAGE",),
                "trimap": ("MASK",),
                "background": (['RGBA', 'BLACK', 'WHITE', 'RED'],),
                "coloradjust": (["enable", "disable"], {"default": "enable"},),            
            },
            "optional":{
                "BgImage_optional": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "remove_background"
    CATEGORY = "AEMatter"

    def remove_background(self, AEMatterModel, image, trimap, background, coloradjust, BgImage_optional=None):
        coloradjust = coloradjust == "enable"
        processed_images = []
        processed_masks = []
        assert background in ['RGBA', 'BLACK', 'WHITE', 'RED']
        for image in image:
            original_image = tensor2numpy(image)
            trimap_image = tensor2numpy(trimap)
            trimap_copy = trimap_image.copy()
            # 获取原始图像尺寸
            height, width, _ = original_image.shape
            new_height = (((height - 1) // 32) + 1) * 32
            new_width = (((width - 1) // 32) + 1) * 32
            pad_height = new_height - height
            pad_height_top = int(pad_height / 2)
            pad_height_bottom = pad_height - pad_height_top
            pad_width = new_width - width
            pad_width_left = int(pad_width / 2)
            pad_width_right = pad_width - pad_width_left

            # 使用 PIL 来填充边界
            padded_image = np.pad(original_image,
                                           ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right), (0, 0)),
                                           mode='reflect')
            padded_trimap = np.pad(trimap_image,
                                            ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)),
                                            mode='reflect')

            # 转换为 numpy 数组
            padded_image_array = np.array(padded_image)
            padded_trimap_array = np.array(padded_trimap)

            # 生成 tritempimgs
            trimap_channels = np.zeros([*padded_trimap_array.shape, 3], np.float32)
            trimap_channels[:, :, 0] = (padded_trimap_array == 0)
            trimap_channels[:, :, 1] = (padded_trimap_array == 128)
            trimap_channels[:, :, 2] = (padded_trimap_array == 255)
            trimap_tensor = np.transpose(trimap_channels, (2, 0, 1))[np.newaxis, :, :, :]

            # 预处理图像
            image_tensor = np.transpose(padded_image_array, (2, 0, 1))[np.newaxis, ::-1, :, :]
            image_tensor = np.array(image_tensor, np.float32) / 255.0
            image_tensor = torch.from_numpy(image_tensor).cuda()
            trimap_tensor = torch.from_numpy(trimap_tensor).cuda()

            # 推理阶段
            with torch.no_grad():
                prediction = AEMatterModel(image_tensor, trimap_tensor)
                prediction = prediction.detach().cpu().numpy()[0]
                prediction = prediction[:, pad_height_top:pad_height_top + height,
                             pad_width_left:pad_width_left + width]
                # mask 矫正
                prediction = np.clip((prediction - 0.1) / (1 - 0.1), 0, 1)
                alpha_prediction = prediction[0:1, ] * 255
                alpha_prediction = np.transpose(alpha_prediction, (1, 2, 0))
                alpha_prediction = alpha_prediction * (trimap_copy[:, :, None] == 128) + (
                            trimap_copy[:, :, None] == 255) * 255

            alpha_prediction = np.array(alpha_prediction, np.uint8)
            if coloradjust:
                original_image = self.adaptive_color_spill_removal(original_image, alpha_prediction)
            pil_im = Image.fromarray(np.squeeze(alpha_prediction))
            if BgImage_optional is not None:
                BgImage_optional = tensor2pil(BgImage_optional)
                BgImage_optional = BgImage_optional.resize((width, height))
                new_im = Image.composite(Image.fromarray(original_image), BgImage_optional, pil_im)
            else:
                if background == "RGBA":
                    new_im = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
                elif background == "BLACK":
                    new_im = Image.new("RGBA", pil_im.size, (0, 0, 0, 255))
                elif background == "WHITE":
                    new_im = Image.new("RGBA", pil_im.size, (255, 255, 255, 255))
                elif background == "RED":
                    new_im = Image.new("RGBA", pil_im.size, (255, 0, 0, 255))
                new_im.paste(Image.fromarray(original_image), mask=pil_im)
            if background != "RGBA":
                new_im = new_im.convert("RGB")

            new_im_tensor = pil2tensor(new_im)  # 将PIL图像转换为Tensor
            pil_im_tensor = pil2tensor(pil_im)  # 同上

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return new_ims, new_masks

    def adaptive_color_spill_removal(self, image, mask):
        # 转换为浮点数类型
        image = image.astype(np.float32)
        mask = mask.astype(np.float32) / 255.0

        # 0-0.8区域膨胀，在该区域进行颜色去溢
        # region_to_dilate = np.where((mask > 0.1) & (mask <= 0.5), 1, 0).astype(np.uint8)
        # k_size = 5
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        # dilated_region = cv2.dilate(region_to_dilate, kernel)


        # 分离通道
        r, g, b = cv2.split(image)

        # 初始化溢色修正图像
        corrected_image = np.zeros_like(image)

        # 计算溢色
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j] > 0 and mask[i, j] < 0.95:
                    # 当前像素的颜色
                    pixel = np.array([r[i, j], g[i, j], b[i, j]])
                    
                    # 计算与其他通道的差异
                    mean_pixel = np.mean(pixel)
                    diff = pixel - mean_pixel
                    
                    # 找到溢色最大的通道
                    spill_channel = np.argmax(diff)
                    
                    # 减弱溢色
                    if diff[spill_channel] > 0:
                    # 减弱溢色
                        pixel[spill_channel] = mean_pixel + (pixel[spill_channel] - mean_pixel) * (1 - mask[i, j])
                        # pixel[spill_channel] = pixel[spill_channel]
                    # 将修正后的颜色赋值回图像
                    corrected_image[i, j] = pixel
                else:
                    corrected_image[i, j] = image[i, j]

        return corrected_image.astype(np.uint8)


class Create_Trimap:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ('MASK', {}),
                "kernel_size": ("INT", {
                    "default": 9,
                    "min": 0,
                    "max": 50,
                    "step": 1
                }),
                "lower_bound": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 127,
                    "step": 1
                }),
                "upper_bound": ("INT", {
                    "default": 200,
                    "min": 128,
                    "max": 255,
                    "step": 1
                }),
            }
        }

    FUNCTION = "createtrimap"
    CATEGORY = "AEMatter"
    RETURN_TYPES = ("MASK",)

    def createtrimap(self, mask, kernel_size, lower_bound, upper_bound):
        res_masks = []
        for mask_ in mask:
            mask_ = mask_.cpu().numpy() * 255
            mask_ = mask_.astype(np.uint8)
            mask_ = self.gen_trimap(mask_, kernel_size=kernel_size, lower_bound=lower_bound,
                                    upper_bound=upper_bound) / 255.

            res_masks.extend([torch.from_numpy(mask_).unsqueeze(0)])
        return (torch.cat(res_masks, dim=0),)

    def gen_trimap(self, alpha, kernel_size=25, lower_bound=50, upper_bound=200):
        k_size = kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        # 对 alpha 进行处理，生成 trimap 区域
        trimap = np.zeros(alpha.shape, dtype=np.uint8)
        trimap.fill(128)
        trimap[alpha < upper_bound] = 0
        trimap[(alpha >= lower_bound) & (alpha <= upper_bound)] = 128
        trimap[alpha > upper_bound] = 255

        # 生成只包含像素值为 128 的 mask
        mask_128 = np.zeros(alpha.shape, dtype=np.uint8)
        mask_128[trimap == 128] = 255

        # 膨胀像素值为 128 的区域
        dilated = cv2.dilate(mask_128, kernel)

        # 将膨胀后的区域与原 trimap 合并
        result = np.where(dilated == 255, 128, trimap)

        return result

NODE_CLASS_MAPPINGS = {
    "AEMatter_ModelLoader": AEMatter_ModelLoader,
    "Create_Trimap": Create_Trimap,
    "AEMatter_Apply": AEMatter_Apply,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AEMatter_ModelLoader": "AEMatter_ModelLoader",
    "Create_Trimap": "Create_Trimap",
    "AEMatter_Apply": "AEMatter_Apply",
}