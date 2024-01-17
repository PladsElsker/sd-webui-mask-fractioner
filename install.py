import launch


if not launch.is_installed('rectpack'):
    launch.run_pip(f'install rectpack', f"rectpack for sd-webui-mask-fractioner")


if not launch.is_installed('sdwi2iextender'):
    launch.run_pip(f'install sdwi2iextender', f"sdwi2iextender for sd-webui-mask-fractioner")
