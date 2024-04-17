import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from PIL import Image
from torchvision import transforms
import torch

class DatasetTDSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H & mask for TDSR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetTDSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H & masks
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        self.paths_M = util.get_image_paths(opt['dataroot_M'])

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def _testval_sync_transform(self, img, mask):
        base_size = self.patch_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(mask,
                                            dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, index):

        L_path = None

        H_path = self.paths_H[index]
        img_H = Image.open(H_path).convert('RGB')

        M_path = self.paths_M[index]
        img_M = Image.open(M_path)

        img_H, img_M = self._testval_sync_transform(img_H, img_M)
        img_L = util.imresize_np(img_H, 1 / self.sf, True)

        if self.opt['phase'] == 'train':
            mode = random.randint(0, 7)
            img_L = util.augment_img(img_L, mode=mode)
            img_H = util.augment_img(img_H, mode=mode)
            img_M = util.augment_img(img_M, mode=mode)

        if L_path is None:
            L_path = H_path

        img_H = util.uint2single(img_H)
        img_H = util.single2tensor3(img_H)
        img_L = util.uint2single(img_L)
        img_L = util.single2tensor3(img_L)

        img_M = np.expand_dims(img_M, axis=0).astype('float32') / 255.0
        img_M = torch.from_numpy(img_M)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path, 'M': img_M}

    def __len__(self):
        return len(self.paths_H)
