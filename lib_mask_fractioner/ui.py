import gradio as gr

from modules.ui_components import InputAccordion

from sdwi2iextender import OperationMode, register_operation_mode
from .globals import MaskFractionerGlobals


class MaskFractionerParams(OperationMode):
    requested_elem_ids = ["img2img_inpaint_full_res", "img2img_inpaint_full_res_padding", "img2img_mask_mode"]

    def section(self, components):
        self.img2img_mask_mode = components["img2img_mask_mode"]
        self.img2img_inpaint_full_res = components["img2img_inpaint_full_res"]
        self.img2img_inpaint_full_res_padding = components["img2img_inpaint_full_res_padding"]

        with InputAccordion(False, label='Mask Fractioner') as self.enabled:
            with gr.Group(visible=False) as self.params_group:
                self.allow_rotations = gr.Checkbox(label='Allow rotations')
                with gr.Row():
                    with gr.Column():
                        self.padding = gr.Slider(label='Inpaint padding', minimum=0, maximum=256, step=4, default=32)
                    with gr.Column():
                        self.margin = gr.Slider(label='Components margin', minimum=0, maximum=10, step=1, default=0)

                with gr.Row():
                    with gr.Column():
                        choices = ["Original", "Unified color"]
                        self.fill_dead_space_with = gr.Radio(label='Fill dead space with', choices=choices, value=choices[0])
                    with gr.Column():
                        self.dead_space_color = gr.ColorPicker(label='Color', visible=False)
    
    def gradio_events(self, *_):
        self._toggle_ui_params()
        self._toggle_only_masked()
        self._toggle_color_picker()
        self._update_globals()

    def _toggle_ui_params(self):
        self.enabled.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=[self.enabled],
            outputs=[self.params_group]
        )

    def _toggle_only_masked(self):
        def handle_only_masked_visibility(enabled):
            choices = [choice_tupple[0] for choice_tupple in self.img2img_inpaint_full_res.choices]
            choice_id = 0 if enabled else getattr(MaskFractionerGlobals, "img2img_inpaint_full_res_restore", 0)
            return gr.update(visible=not enabled, value=choices[choice_id]), gr.update(visible=not enabled)

        self.enabled.change(
            fn=handle_only_masked_visibility,
            inputs=[self.enabled],
            outputs=[self.img2img_inpaint_full_res, self.img2img_inpaint_full_res_padding]
        )

        self.img2img_inpaint_full_res.input(
            fn=lambda value: setattr(MaskFractionerGlobals, "img2img_inpaint_full_res_restore", value),
            inputs=[self.img2img_inpaint_full_res],
            outputs=[]
        )

    def _toggle_color_picker(self):
        SHOW_COLOR_PICKER_VALUE = "Unified color"
        self.fill_dead_space_with.change(
            fn=lambda radio_value: gr.update(visible=radio_value==SHOW_COLOR_PICKER_VALUE),
            inputs=[self.fill_dead_space_with],
            outputs=[self.dead_space_color]
        )

    def _update_globals(self):
        set_global_value_dict = dict(
            fn=lambda k, v: setattr(MaskFractionerGlobals, k, v),
            outputs=[]
        )
        
        self.img2img_mask_mode.change(**set_global_value_dict, inputs=[gr.State("img2img_mask_mode"), self.img2img_mask_mode])
        self.enabled.change(**set_global_value_dict, inputs=[gr.State("enabled"), self.enabled])
        self.padding.change(**set_global_value_dict, inputs=[gr.State("padding"), self.padding])
        self.padding.release(**set_global_value_dict, inputs=[gr.State("padding"), self.padding])
        self.margin.change(**set_global_value_dict, inputs=[gr.State("margin"), self.margin])
        self.margin.release(**set_global_value_dict, inputs=[gr.State("margin"), self.margin])
        self.fill_dead_space_with.change(**set_global_value_dict, inputs=[gr.State("fill_dead_space_with"), self.fill_dead_space_with])
        self.dead_space_color.change(**set_global_value_dict, inputs=[gr.State("dead_space_color"), self.dead_space_color])
        self.allow_rotations.change(**set_global_value_dict, inputs=[gr.State("allow_rotations"), self.allow_rotations])


def register_inpaint_params():
    register_operation_mode(MaskFractionerParams)
