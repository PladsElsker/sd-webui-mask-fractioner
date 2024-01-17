import torch

import modules.scripts as scripts
import gradio as gr

from lib_mask_fractioner.fracturing import fracture_images, emplace_back_images
from lib_mask_fractioner.webui_callbacks import setup_script_callbacks
from lib_mask_fractioner.globals import MaskFractionerGlobals


class MaskFractionerScript(scripts.Script):
    def __init__(self):
        self.unfractioned_images = None
        self.mask = None
        self.rearranged_image_data = None

    def title(self):
        return 'Mask Fractioner'

    def ui(self, is_img2img):
        return []

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def before_process(self, p, *_):
        if not MaskFractionerGlobals.enabled:
            return

        self.unfractioned_images = p.init_images
        self.mask = p.image_mask

        if not self._is_mask_valid():
            return

        should_minimize_dead_space = MaskFractionerGlobals.fill_dead_space_with=="Original"

        self.rearranged_image_data = fracture_images(
            images=p.init_images,
            mask=p.image_mask,
            padding=MaskFractionerGlobals.padding,
            components_margin=MaskFractionerGlobals.margin,
            allow_rotations=MaskFractionerGlobals.allow_rotations,
            dead_space_color=MaskFractionerGlobals.dead_space_color,
            minimize_dead_space=should_minimize_dead_space
        )
        p.init_images, p.image_mask = self.rearranged_image_data.images, self.rearranged_image_data.mask

    def postprocess_batch_list(self, p, pp, *_, **__):
        if not MaskFractionerGlobals.enabled:
            return
        
        if not self._is_mask_valid():
            return

        processed = emplace_back_images(
            arrangement_panel=self.rearranged_image_data.panel,
            original_images=self.unfractioned_images,
            inpainted_images=pp.images,
            mask_blur=p.mask_blur
        )

        pp.images.clear()
        pp.images.extend([torch.tensor(img.transpose(2, 0, 1) / 255) for img in processed])

        p.overlay_images = []
    

    def _is_mask_valid(self):
        if self.mask is None:
            return False
        
        min_value_mask, max_value_mask = self.mask.getextrema()
        if min_value_mask == max_value_mask == 0:
            return False

        return True


setup_script_callbacks()
