import numpy as np
import torch

import modules.scripts as scripts
import gradio as gr

from lib_mask_fractioner.fracturing import fracture_images, emplace_back_images
from lib_mask_fractioner.webui_callbacks import setup_script_callbacks


class MaskFractionerScript(scripts.Script):
    def __init__(self):
        self.unfractioned_images = None
        self.mask = None
        self.rearranged_image_data = None

    def title(self):
        return 'Mask Fractioner'

    def ui(self, is_img2img):
        with gr.Accordion('Mask Fractioner', open=False):
            enabled = gr.Checkbox(label='Enabled', default=False)
            with gr.Group(visible=False) as params_group:
                with gr.Row():
                    with gr.Column():
                        padding = gr.Slider(label='Inpaint padding per component', minimum=0, maximum=256, step=4,
                                            default=32)
                    with gr.Column():
                        components_margin = gr.Slider(label='Components margin', minimum=0, maximum=10, step=1, default=0)

                with gr.Row():
                    with gr.Column():
                        dead_space_color = gr.ColorPicker(label='Fill dead space with')
                    with gr.Column():
                        allow_rotations = gr.Checkbox(label='Allow rotations')

            enabled.change(lambda enabled_state: gr.update(visible=enabled_state), inputs=[enabled], outputs=[params_group])

        return [enabled, padding, components_margin, allow_rotations, dead_space_color]

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def before_process(self, p, enabled, padding, components_margin, allow_rotations, dead_space_color):
        if not enabled:
            return

        self.unfractioned_images = p.init_images
        self.mask = p.image_mask


        if self.mask is None:
            return

        self.rearranged_image_data = fracture_images(
            images=p.init_images,
            mask=p.image_mask,
            padding=padding,
            components_margin=components_margin,
            allow_rotations=allow_rotations,
            dead_space_color=dead_space_color,
        )
        p.init_images, p.image_mask = self.rearranged_image_data.images, self.rearranged_image_data.mask

    def postprocess_batch_list(self, p, pp, enabled, padding, components_margin, allow_rotations, dead_space_color, **kwargs):
        if not enabled:
            return

        if self.mask is None:
            return

        processed = emplace_back_images(
            arrangement_panel=self.rearranged_image_data.panel,
            original_images=self.unfractioned_images,
            inpainted_images=pp.images,
            p=p
        )

        pp.images.clear()
        pp.images.extend([torch.tensor(img.transpose(2, 0, 1) / 255) for img in processed])

        p.overlay_images = []


setup_script_callbacks()
