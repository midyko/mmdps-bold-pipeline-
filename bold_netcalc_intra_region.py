import os
import numpy as np
import nibabel as nib

from mmdps.proc import atlas
# from mmdps.util.loadsave import load_nii, save_csvmat
from mmdps.util import path
from mmdps.proc import job

if __name__ == '__main__':
	
	atlasobj = path.curatlas()
	volumename = '3mm'
	print(os.path.join(path.curparent(), 'pBOLD.nii'))
	print(os.getcwd())
	
	outfolder = os.path.join(os.getcwd(), 'bold_net_attr_zzl')
	print(outfolder)

