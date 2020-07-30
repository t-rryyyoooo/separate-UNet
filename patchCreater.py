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

        model.eval()

        # Read image.
        self.image = sitk.ReadImage(self.image_path)
        print("Image shape : {}".format(self.image.GetSize()))

        # Pad or crop the image in sagittal and coronal direction to plane_size. 
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

        print("Image shape : {}".format(self.image.GetSize()))

        # Pad the image in axial direction to just make patches.
        image_shape = np.array(self.image.GetSize())
        slide = self.input_size // np.array((1, 1, self.overlap))
        lower_pad_size, upper_pad_size = caluculatePaddingSize(image_shape, self.input_size, self.input_size, slide)

        padded_image = padding(self.image, lower_pad_size[0].tolist(), upper_pad_size[0].tolist())

        padded_image_array = sitk.GetArrayFromImage(padded_image)
        print("Padded image shape : {}".format(padded_image.GetSize()))

        padded_image_shape = np.array(padded_image.GetSize()) - self.input_size
        self.feature_map_list= []
        length = (padded_image_shape[2] // slide[2]) + 1
        with tqdm(total=length, desc="Making feature maps...", ncols=60) as pbar:
            for z in range(0, padded_image_shape[2] + 1, slide[2]):
                z_slice = slice(z, z + self.input_size[2])
                batch = padded_image[:, :, z_slice]
                batch_array = sitk.GetArrayFromImage(batch)
                batch_array = torch.from_numpy(batch_array).to(device, dtype=torch.float)
                batch_array = batch_array[None, None, ...]

                for c in range(self.output_layer):
                    batch_array, feature_map = model.contracts[c](batch_array)

                feature_map = feature_map.to("cpu").detach().numpy()
                feature_map = np.squeeze(feature_map)

                self.feature_map_list.append(feature_map)
                pbar.update(1)

    def output(self):
        return self.patch_array_list

    def save(self, save_path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with tqdm(total=len(self.feature_map_list), desc="Saving images...", ncols=60) as pbar:
            for i, feature_map in enumerate(self.feature_map_list):
                path = save_path / ("image_" + str(i).zfill(3) + ".npy")
                np.save(str(path), feature_map)

                pbar.update(1)


