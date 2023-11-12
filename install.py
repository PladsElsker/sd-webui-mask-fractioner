import launch


if not launch.is_installed('rectpack'):
    launch.run_pip(f'install rectpack', f"rectpack for sd-webui-mask-fractioner")
