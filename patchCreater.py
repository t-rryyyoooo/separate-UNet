import SimpleITK as sitk
import cloudpickle
import numpy as np
import torch
from pathlib import Path
from functions import caluculatePaddingSize, cropping, rounding, padding, getImageWithMeta
from tqdm import tqdm

class PatchCreater():
    def __init__(self, image_path, modelweight_path, output_layer=1, input_size=(32, 32, 32), plane_size=(512, 512), overlap=1, gpu_ids=[0]):
        """
        simpleitk shape : sagittal, coronal, axial
        numpy shape : axial coronal, sagittal
        input_size : sagittal, coronal, axial
        """
        self.image_path = image_path
        self.modelweight_path = modelweight_path
        self.output_layer = output_layer
        self.input_size = np.array(input_size)
        self.plane_size = np.array(plane_size)
        self.overlap = overlap
        self.gpu_ids = gpu_ids


    def execute(self):
        is_cuda = torch.cuda.is_available() and True
        device = torch.device("cuda" if is_cuda else "cpu")

        # Load model.
        with open(self.modelweight_path, "rb") as f:
            model = cloudpickle.load(f)
#model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)

        model.eval()

        # Pad an image considering stride etc. 
        self.image = sitk.ReadImage(self.image_path)

        required_shape = np.array(self.image.GetSize())
        required_shape[0:2] = self.plane_size

        diff = required_shape - np.array(self.image.GetSize())
        if (diff < 0).any():
            lower_crop_size = (abs(diff) // 2).tolist()
            upper_crop_size = [rounding(x, 1) for x in abs(diff) / 2]

            self.image = cropping(self.image, lower_crop_size, upper_crop_size)

        else:
            lower_pad_size = (diff // 2).tolist()
            upper_pad_size = [rounding(x, 1) for x in diff / 2]

            self.image = padding(self.image, lower_pad_size, upper_pad_size)

        slide = self.input_size // np.array((1, 1, self.overlap))
        lower_pad_size, upper_pad_size = caluculatePaddingSize(np.array(self.image.GetSize()), self.input_size, self.input_size, slide)

        padded_image = padding(self.image, lower_pad_size[0].tolist(), upper_pad_size[0].tolist(), mirroring=True)

        padded_image_array = sitk.GetArrayFromImage(padded_image)

        z_size, y_size, x_size = padded_image_array.shape
        
        self.patch_array_list = []
        length = (z_size // slide[2]) * (y_size // slide[1]) * (x_size // slide[0])
        with tqdm(total=length, desc="Clipping and concatenating images...", ncols=60) as pbar:
            for z in range(0, z_size, slide[2]):
                patch_array = [] # For one patch
                for y in range(0, y_size, slide[1]):
                    batch_array = [] # To output feature maps for clipped images at once.
                    for x in range(0, x_size, slide[0]):
                        z_slice = slice(z, z + self.input_size[2])
                        y_slice = slice(y, y + self.input_size[1])
                        x_slice = slice(x, x + self.input_size[0])

                        # Make mini patch.
                        mini_patch_array = padded_image_array[z_slice, y_slice, x_slice]
                        batch_array.append(mini_patch_array)
                        pbar.update(1)

                    batch_array = np.stack(batch_array)
                    batch_array = torch.from_numpy(batch_array).to(device, dtype=torch.float)
                    batch_array = batch_array[:, None, ...]

                    for c in range(self.output_layer):
                        batch_array, feature_maps = model.contracts[c](batch_array)

                    feature_maps = feature_maps.to("cpu").detach().numpy()
                    # Concat images in sagittal direction. make retangular-solid.
                    feature_maps = np.concatenate([feature_maps[x, ...] for x in range(x_size // slide[0])], axis=-1)

                    patch_array.append(feature_maps)

                # Concat images in coronal direction. Make original size in sagittal and coronal direction, square.
                patch_array = np.concatenate(patch_array, axis=-2)

                self.patch_array_list.append(patch_array)

    def output(self):
        return self.patch_array_list

    def save(self, save_path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with tqdm(total=len(self.patch_array_list), desc="Saving images...", ncols=60) as pbar:
            for i, patch in enumerate(self.patch_array_list):
                path = save_path / ("image_" + str(i).zfill(3) + ".npy")
                np.save(str(path), patch)

                pbar.update(1)


