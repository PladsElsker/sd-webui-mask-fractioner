class WebuiInpaintRatioHook:
    ratio = None

    @classmethod
    def on_after_component(cls, component, **kwargs):
        elem_id = kwargs.get('elem_id', None)

        if elem_id is None:
            return

        if elem_id == 'img2img_width':
            cls.width_slider = component
            cls.resize_to_tab = component.parent.parent.parent

        if elem_id == 'img2img_height':
            cls.height_slider = component
            cls.ratio = cls.width_slider.value / cls.height_slider.value

        if elem_id == 'img2img_scale':
            cls.scale_slider = component
            cls.resize_by_tab = component.parent

            cls.setup_gradio_events()

    @classmethod
    def setup_gradio_events(cls):
        def set_ratio(ratio):
            cls.ratio = ratio

        resize_to_event_params = {
            'fn': lambda x, y: set_ratio(x/y),
            'inputs': [cls.width_slider, cls.height_slider],
            'outputs': [],
        }

        cls.resize_to_tab.select(**resize_to_event_params)
        cls.width_slider.release(**resize_to_event_params)
        cls.height_slider.release(**resize_to_event_params)

        resize_by_event_params = {
            'fn': lambda: set_ratio(None),
            'inputs': [],
            'outputs': [],
        }

        cls.resize_by_tab.select(**resize_by_event_params)
        cls.scale_slider.release(**resize_by_event_params)
