import launch
import pkg_resources


if not launch.is_installed('rectpack'):
    launch.run_pip(f'install rectpack', f"rectpack for sd-webui-mask-fractioner")


minimum_sdwi2iextender_version = "0.1.2"
if not launch.is_installed('sdwi2iextender'):
    launch.run_pip(f'install sdwi2iextender', f"sdwi2iextender for sd-webui-mask-fractioner")
else:
    current_version = pkg_resources.get_distribution("sdwi2iextender").version
    if pkg_resources.parse_version(current_version) < pkg_resources.parse_version(minimum_sdwi2iextender_version):
        launch.run_pip(f'install sdwi2iextender=={minimum_sdwi2iextender_version}', f"sdwi2iextender=={minimum_sdwi2iextender_version} for sd-webui-mask-fractioner")
