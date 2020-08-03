import SimpleITK as sitk
import cloudpickle
import torch
from functions import resampleSize, cropping, rounding, padding, caluculatePaddingSize
import numpy as np
from tqdm import tqdm
from pathlib import Path

class ThinPatchCreater():
    def __init__(self, image, model_path, label_patch_size, plane_size, overlap, num_down, is_label=False):
        self.image = image
        self.model_path = model_path
        self.label_patch_size = np.array(label_patch_size)
        self.plane_size = np.array(plane_size)
        self.overlap = overlap
        self.num_down = num_down
        self.is_label = is_label

    def execute(self):
        # Crop or pad the image for required_shape.
        image = self.image
        print("Image shape : {}".format(image.GetSize()))

        image_shape = np.array(image.GetSize())
        required_shape = np.array(image.GetSize())
        required_shape[0:2] = self.plane_size
        diff = required_shape - image_shape
        if (diff < 0).any():
            lower_crop_size = (abs(diff) // 2).tolist()
            upper_crop_size = [rounding(x, 1) for x in abs(diff) / 2]

            image = cropping(image, lower_crop_size, upper_crop_size)

        else:
            lower_pad_size = (diff // 2).tolist()
            upper_pad_size = [rounding(x, 1) for x in diff / 2]

            image = padding(image, lower_pad_size, upper_pad_size)

        print("Image shape : {}".format(image.GetSize()))

        image_shape = np.array(image.GetSize())
        slide = self.label_patch_size // np.array((1, 1, self.overlap))
        lower_pad_size, upper_pad_size = caluculatePaddingSize(image_shape, self.label_patch_size, self.label_patch_size, slide)
        image = padding(image, lower_pad_size[0].tolist(), upper_pad_size[0].tolist())
        print("Image shape : {}".format(image.GetSize()))

        # Downsample the image to one in num_down.
        required_shape = np.array(image.GetSize()) // 2**self.num_down
        if not self.is_label:
            image = resampleSize(image, required_shape.tolist(), is_label=False)
        else:
            image = resampleSize(image, required_shape.tolist(), is_label=True)

        print("Image shape : {}".format(image.GetSize()))
        # Crop the image to (label_patch_size / num_down)
        image_shape = np.array(image.GetSize())
        patch_size = self.label_patch_size // 2**self.num_down
        _, _, z_length = image_shape - patch_size
        slide = patch_size // np.array((1, 1, self.overlap))
        total = z_length // slide[2] + 1
        self.patch_list = []
        self.patch_array_list = []
        with tqdm(total=total, desc="Clipping images...", ncols=60) as pbar:
            for z in range(0, z_length + 1, slide[2]):
                z_slice = slice(z, z + patch_size[2])
                patch = image[:, :, z_slice]
                patch.SetOrigin(self.image.GetOrigin())

                patch_array = sitk.GetArrayFromImage(patch)

                self.patch_list.append(patch)
                self.patch_array_list.append(patch_array)
                
                pbar.update(1)

        # Load model
        with open(self.model_path, "rb") as f:
            model = cloudpickle.load(f)
        model.eval()

        # Make feature map.
        total = len(self.patch_array_list)
        is_cuda = torch.cuda.is_available() and True
        device = torch.device("cuda" if is_cuda else "cpu")
        self.feature_map_list = []
        with tqdm(total = total, desc="Making feature maps...", ncols=60) as pbar:
            for patch_array in self.patch_array_list:
                patch_array = torch.from_numpy(patch_array).to(device, dtype=torch.float)[None, None,...]
                feature_map = model.forwardWithoutSegmentation(patch_array)
                feature_map = feature_map.to("cpu").detach().numpy()
                feature_map = np.squeeze(feature_map)
                self.feature_map_list.append(feature_map)

                pbar.update(1)


    def save(self, save_path, kind):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        if not self.is_label:
            file_name = "image"
        else:
            file_name = "label"

        if kind == "Array":
            length = len(self.patch_array_list)
            with tqdm(total=length, desc="Saving image arrays...", ncols=60) as pbar:
                for i, patch_array in enumerate(self.patch_array_list):
                    path = save_path / "{}_{}.npy".format(file_name, str(i).zfill(3))
                    np.save(str(path), patch_array)
                    pbar.update(1)

        elif kind == "Image":
            length = len(self.patch_list)
            with tqdm(total=length, desc="Saving images...", ncols=60) as pbar:
                for i, patch in enumerate(self.patch_list):
                    path = save_path / "{}_{}.mha".format(file_name, str(i).zfill(3))
                    sitk.WriteImage(patch, str(path), True)
                    pbar.update(1)

        elif kind == "Feature_map":
            length = len(self.feature_map_list)
            with tqdm(total=length, desc="Saving feature maps...", ncols=60) as pbar:
                for i, feature_map in enumerate(self.feature_map_list):
                    path = save_path / "{}_{}.npy".format(file_name, str(i).zfill(3))
                    np.save(str(path), feature_map)
                    pbar.update(1)


        else:
            print("[ERROR] Kind must be Array/Image/Feature_map.")
            sys.exit()


