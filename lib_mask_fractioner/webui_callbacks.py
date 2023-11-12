from modules.scripts import script_callbacks
from lib_mask_fractioner.webui_inpaint_ratio_hook import WebuiInpaintRatioHook


def setup_script_callbacks():
    script_callbacks.on_after_component(WebuiInpaintRatioHook.on_after_component)
