import math
from dataclasses import dataclass
from typing import Sequence, Optional

import cv2
import numpy as np
from PIL import Image, ImageChops

from rectpack import newPacker, PackingBin

from lib_mask_fractioner.webui_inpaint_ratio_hook import WebuiInpaintRatioHook
from lib_mask_fractioner.mask_processing import blur, mask_interpolation


@dataclass
class MaskData:
    mask: np.array
    offset: Sequence[int] = (0, 0)
    rotated: bool = False

    @property
    def min_x(self):
        return self.offset[1]

    @property
    def min_y(self):
        return self.offset[0]

    @property
    def max_x(self):
        return self.offset[1] + self.mask.shape[1]

    @property
    def max_y(self):
        return self.offset[0] + self.mask.shape[0]

    @property
    def area(self):
        return slice(self.min_y, self.max_y), slice(self.min_x, self.max_x)

    @property
    def size(self):
        return self.mask.shape


@dataclass
class MaskPanel:
    component_masks: Sequence[MaskData]
    cropped_masks: Optional[Sequence[MaskData]] = None
    rearranged_masks: Optional[Sequence[MaskData]] = None
    size: Optional[Sequence[int]] = None


@dataclass
class RearrangedImageData:
    panel: MaskPanel
    images: Sequence
    mask: Image


def fracture_images(images, mask, padding, components_margin, allow_rotations, dead_space_color, minimize_dead_space, invert_mask):
    mask_panel = compute_mask_cca(mask)
    compute_crops(mask_panel, padding)
    compute_rearrangement(mask_panel, components_margin, allow_rotations)
    rearranged_image_data = rearrange_images_and_mask(mask_panel, images, dead_space_color)
    if invert_mask:
        rearranged_image_data.mask = ImageChops.invert(rearranged_image_data.mask)
    
    return rearranged_image_data


def compute_mask_cca(mask):
    image_array = np.array(mask.convert('L'))
    binary_image = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    num_masks, labels = cv2.connectedComponents(binary_image)

    component_masks = []
    for i in range(num_masks):
        # skip backgrounds
        if binary_image[labels == i][0] == 0:
            continue

        component_mask = np.zeros_like(binary_image, dtype=np.uint8)
        component_mask[labels == i] = 255
        mask_data = MaskData(component_mask)
        component_masks.append(mask_data)

    return MaskPanel(component_masks)


def compute_crops(mask_panel, padding):
    crops = []

    def clamp(lower_bound, upper_bound, val):
        if val < lower_bound:
            return lower_bound
        if val > upper_bound:
            return upper_bound
        return val

    for component_data in mask_panel.component_masks:
        bbox = find_bounding_box(component_data.mask)
        cropped_mask_no_padding = component_data.mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        cropped_mask = np.pad(
            cropped_mask_no_padding,
            (
                (clamp(0, padding, bbox[0]), clamp(0, padding, component_data.mask.shape[0] - bbox[2])),
                (clamp(0, padding, bbox[1]), clamp(0, padding, component_data.mask.shape[1] - bbox[3])),
            ),
            constant_values=0
        )
        cropped_data = MaskData(cropped_mask, offset=[
            bbox[0] - clamp(0, padding, bbox[0]),
            bbox[1] - clamp(0, padding, bbox[1])
        ])
        crops.append(cropped_data)

    mask_panel.cropped_masks = crops


def find_bounding_box(mask):
    nonzero_indices = np.where(mask > 0)

    min_row, min_col = np.min(nonzero_indices[0]), np.min(nonzero_indices[1])
    max_row, max_col = np.max(nonzero_indices[0]), np.max(nonzero_indices[1])

    return min_row, min_col, max_row, max_col


def compute_rearrangement(mask_panel, components_margin, allow_rotations, grow_factor=1.1):
    rectangles = [[d + components_margin for d in cropped.size] for cropped in mask_panel.cropped_masks]

    total_area = sum(w * h for w, h in rectangles)
    bin_width = math.sqrt(total_area * WebuiInpaintRatioHook.ratio)

    bin_height = total_area / bin_width
    packed = False
    packer = None

    while not packed:
        packer = newPacker(bin_algo=PackingBin.BFF, rotation=allow_rotations)
        for i, r in enumerate(rectangles):
            packer.add_rect(r[1], r[0], rid=i)
        packer.add_bin(int(bin_width), int(bin_height))

        packer.pack()

        if len(packer) > 0 and len(packer[0]) == len(rectangles):
            packed = True
        else:
            bin_width = grow_factor * bin_width
            bin_height = grow_factor * bin_height

    packed_rectangles = packer.rect_list()
    packed_rectangles.sort(key=lambda rect_data: rect_data[5])
    rearranged_masks = []
    max_x = 0
    max_y = 0
    for cropped_data, rect in zip(mask_panel.cropped_masks, packed_rectangles):
        b, x, y, w, h, rid = rect
        max_x = max(x+w, max_x)
        max_y = max(y+h, max_y)

        mod_cropped_mask = cropped_data.mask
        is_rotated = w - components_margin == cropped_data.mask.shape[0] and w != h
        if is_rotated:
            mod_cropped_mask = np.rot90(cropped_data.mask)

        rearranged_data = MaskData(mod_cropped_mask, offset=[
            y + components_margin // 2,
            x + components_margin // 2
        ], rotated=is_rotated)
        rearranged_masks.append(rearranged_data)

    mask_panel.rearranged_masks = rearranged_masks

    bin_error_x = (bin_width - max_x) / bin_width
    bin_error_y = (bin_height - max_y) / bin_height

    if bin_error_x < bin_error_y:
        bin_width = int(bin_width / (1 + bin_error_x))
        bin_height = int(bin_height / (1 + bin_error_x))
    else:
        bin_width = int(bin_width / (1 + bin_error_y))
        bin_height = int(bin_height / (1 + bin_error_y))

    mask_panel.size = (bin_height, bin_width)


def rearrange_images_and_mask(mask_panel, images, dead_space_color):
    dead_space_color = np.array([int(dead_space_color[i:i+2], 16) for i in (1, 3, 5)])
    rearranged_images = []

    for img in images:
        img = np.array(img.convert('RGB'))
        rearranged = np.full([mask_panel.size[0], mask_panel.size[1], 3], dead_space_color).astype(np.uint8)
        for initial, target in zip(mask_panel.cropped_masks, mask_panel.rearranged_masks):
            if target.rotated:
                rearranged[target.area] = np.rot90(img[initial.area])
            else:
                rearranged[target.area] = img[initial.area]

        rearranged_images.append(Image.fromarray(rearranged).convert('RGB'))

    rearranged = np.full(mask_panel.size, 0).astype(np.uint8)
    for target in mask_panel.rearranged_masks:
        rearranged[target.area] = target.mask[:, :]

    rearranged_mask = Image.fromarray(rearranged).convert('L')

    return RearrangedImageData(mask_panel, rearranged_images, rearranged_mask)


def emplace_back_images(arrangement_panel: MaskPanel, original_images, inpainted_images, mask_blur):
    rearranged_images = []

    for img, inpainted in zip([original_images[0]]*len(inpainted_images), inpainted_images):
        img = np.array(img.convert('RGB'))
        inpainted = (inpainted.permute(1, 2, 0).cpu() * 255).numpy().astype(np.uint8)
        inpainted = cv2.resize(inpainted, (arrangement_panel.size[1], arrangement_panel.size[0]))

        for initial, target in zip(arrangement_panel.cropped_masks, arrangement_panel.rearranged_masks):
            inpaint_region = np.rot90(inpainted[target.area], k=-1) if target.rotated else inpainted[target.area]
            mask_region = np.rot90(target.mask, k=-1) if target.rotated else target.mask
            mask_region = blur(mask_region, mask_blur)
            inpaint_region = mask_interpolation(mask_region, img[initial.area], inpaint_region).astype(np.uint8)
            img[initial.area] = inpaint_region

        rearranged_images.append(img)

    return rearranged_images
