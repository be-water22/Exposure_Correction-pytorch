'''
@Autuor: LZ-CH
@Contact: 2443976970@qq.com

Note: this repository could only be used when CUDA is available!!!
'''

import torch
import torch.nn as nn
import torchvision
import os
import time
import model
import numpy as np
import glob
import time
import cv2
from tools.decomposition import lplas_decomposition as decomposition
from model import MSPEC_Net
from tools.calculate_psnr_ssim import calculate_psnr_ssim
from tiler import Tiler, Merger
import random
from tiling import ConstSizeTiles
os.environ['CUDA_VISIBLE_DEVICES']='0'
def exposure_correction(MSPEC_net,data_input):
	if data_input.dtype == 'uint8':
		data_input = data_input/255
	_,L_list = decomposition(data_input)
	L_list = [torch.from_numpy(data).float().permute(2,0,1).unsqueeze(0).cuda() for data in L_list]
	Y_list = MSPEC_net(L_list)
	predict = Y_list[-1].squeeze().permute(1,2,0).detach().cpu().numpy()
	return predict
			
def down_correction(MSPEC_net,data_input):
	maxsize = max([data_input.shape[0],data_input.shape[1]])
	insize = 512
	
	scale_ratio = insize/maxsize
	im_low = cv2.resize(data_input,(0, 0), fx=scale_ratio, fy=scale_ratio,interpolation = cv2.INTER_CUBIC)
	top_pad,left_pad = insize - im_low.shape[0],insize - im_low.shape[1]
	im = cv2.copyMakeBorder(im_low, top_pad, 0, left_pad, 0, cv2.BORDER_DEFAULT)
	out = exposure_correction(MSPEC_net,im)
	out = out[top_pad:,left_pad:,:]
	final_out = out

	'''
	A simple upsampling method is used here. 
	If you want to achieve better results, please 
	use the bgu in the original matlab code to upsample.
	'''

	final_out = cv2.resize(final_out,(data_input.shape[1],data_input.shape[0]))

	return final_out
		


def evaluate(MSPEC_net,image_path,savedir):

	data_input = cv2.imread(image_path)

	start = time.time()
	output_image = down_correction(MSPEC_net,data_input)
	end_time = (time.time() - start)
	image_basename=os.path.basename(image_path)
	if output_image.dtype == 'uint8':
		cv2.imwrite( os.path.join(savedir,image_basename),output_image)
	else:
		cv2.imwrite( os.path.join(savedir,image_basename),output_image*255)


if __name__ == '__main__':
# test_images
	print('-------begin test--------')
	with torch.no_grad():
		MSPEC_net = MSPEC_Net().cuda()
		MSPEC_net =torch.nn.DataParallel(MSPEC_net)
		MSPEC_net.load_state_dict(torch.load('./snapshots/MSPECnet_woadv.pth'))
		MSPEC_net.eval()
		filedir ='./MultiExposure_dataset/testing/INPUT_IMAGES'
		gtimg_dir = './MultiExposure_dataset/testing/expert_c_testing_set'
		test_list = glob.glob(filedir+"/*") 
		test_list.sort()
		savedir = './MultiExposure_dataset/testing/eval_output'
		if not os.path.exists(savedir):
    			os.makedirs(savedir)
		for n,imagepath in enumerate (test_list):
			evaluate(MSPEC_net,imagepath,savedir)
			if ((n+1)%100 == 0):
				print(n+1)
			# if n+1 == 500:
			# 	print('calculate_psnr_ssim:')
			# 	psnr,ssim = calculate_psnr_ssim(savedir,gtimg_dir)
			# 	break
		calculate_psnr_ssim(savedir,gtimg_dir)


"""
'''@Author: LZ-CH
@Contact: 2443976970@qq.com
Note: this repository could only be used when CUDA is available!!!
'''

import os
import time
environment = os.environ
environment['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import glob
import torch
import numpy as np
from model import MSPEC_Net
from tools.decomposition import lplas_decomposition as decomposition
from tools.calculate_psnr_ssim import calculate_psnr_ssim

# Import BGU upsampling utilities
from bgu import compute_bgu, rgb2luminance

# --- Configuration ---
IN_SIZE = 512   # patch size for network input
BGU_THRESH = IN_SIZE  # use BGU if max dimension > this


def exposure_correction(mspec_net, img):
    
    # Perform exposure correction at a fixed resolution patch.
    # Input: img as HxWxC float in [0,1] or uint8.
    # Returns: corrected image at same size.
    
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    _, L_list = decomposition(img)
    # Prepare pyramid tensors
    L_list = [torch.from_numpy(l).float().permute(2,0,1).unsqueeze(0).cuda()
              for l in L_list]
    # Network prediction
    Y_list = mspec_net(L_list)
    out = Y_list[-1].squeeze().permute(1,2,0).detach().cpu().numpy()
    return out


def down_correction(mspec_net, img):
    
    # Downscale large inputs to IN_SIZE, apply exposure correction, then upsample
    # using BGU if high-res, otherwise simple resize.
    
    h, w = img.shape[:2]
    max_dim = max(h, w)
    scale = IN_SIZE / max_dim

    # Low-res input
    img_low = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    # Pad to square resolution IN_SIZE x IN_SIZE
    top_pad = IN_SIZE - img_low.shape[0]
    left_pad = IN_SIZE - img_low.shape[1]
    img_padded = cv2.copyMakeBorder(img_low, top_pad, 0, left_pad, 0,
                                    cv2.BORDER_REFLECT)

    # Exposure correction at low-res
    corrected = exposure_correction(mspec_net, img_padded)
    # Crop back
    corrected = corrected[top_pad:, left_pad:, :]

    # If image is large, apply BGU guided upsampling
    if max_dim > BGU_THRESH:
        # Normalize to float64 in [0,1]
        I_lr = img_low.astype(np.float64) / 255.0
        O_lr = corrected.astype(np.float64)
        I_hr = img.astype(np.float64) / 255.0

        # compute BGU result
        bgu_res = compute_bgu(
            I_lr, rgb2luminance(I_lr),
            O_lr, None,
            I_hr, rgb2luminance(I_hr)
        )
        upsampled = bgu_res['result_fs']
    else:
        # simple interpolation
        upsampled = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_CUBIC)

    # Convert back to uint8
    out_uint8 = np.clip(upsampled*255.0, 0, 255).astype(np.uint8)
    return out_uint8


def evaluate(mspec_net, image_path, save_dir):
    img = cv2.imread(image_path)
    start = time.time()
    out = down_correction(mspec_net, img)
    elapsed = time.time() - start
    print(f"Processed {os.path.basename(image_path)} in {elapsed:.2f}s")

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), out)


if __name__ == '__main__':
    print('------- begin test --------')
    # load network
    net = MSPEC_Net().cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load('./snapshots/MSPECnet_woadv.pth'))
    net.eval()

    input_dir = './MultiExposure_dataset/testing/INPUT_IMAGES'
    gt_dir    = './MultiExposure_dataset/testing/expert_c_testing_set'
    output_dir= './MultiExposure_dataset/testing/eval_output_bgu'

    img_list = sorted(glob.glob(os.path.join(input_dir, '*')))

    with torch.no_grad():
        for idx, path in enumerate(img_list, 1):
            evaluate(net, path, output_dir)
            if idx % 50 == 0:
                print(f"{idx} images done")

    # compute metrics
    calculate_psnr_ssim(output_dir, gt_dir)

"""
