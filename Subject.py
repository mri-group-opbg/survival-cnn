import nibabel as nb
import numpy as np
import imageio
from PIL import Image

class Subject():
    
    def __init__(self, subject_dir, subject):
        self.subject_dir = subject_dir
        self.subject = subject

    """
    get_sequence: returns a sequence given a file for a sequence (e.g. T1, T2, ADC...)
    """
    def get_sequence(self, sequence):
        return nb.load(f"{self.subject_dir}/{self.subject}/{sequence}_registered.nii")
    
    """
    get_roi: returns a roi given a a roi type (e.g. SOLID)
    """
    def get_roi(self, roi_type):
        return nb.load(f"{self.subject_dir}/{self.subject}/ROI/{roi_type}.nii")
    
    """
    get_roi_size: return the size of a roi given along a given axis
    """
    def get_roi_size(self, roi_type, axis):
        roi = self.get_roi(roi_type)
        roi_data = roi.get_fdata()
        
        other_axis = list(range(len(roi_data.shape)))
        other_axis.remove(axis)

        return np.sum(roi_data, axis=tuple(other_axis))

    """
    get_roi_index: return the index of non empty roi along an axis
    """
    def get_roi_index(self, roi_type, axis):
        return np.where(self.get_roi_size(roi_type, axis) > 0)[0]

    """
    get_roi_index_percentile: return the indexes of roi along a given axis
    that are above a given percentile (only non empty roi are considered for
    computation of percentile)
    """
    def get_roi_index_percentile(self, roi_type, axis, percentile):
        roi_sizes = self.get_roi_size(roi_type, axis)
        
        non_empty_sizes = roi_sizes[np.where(roi_sizes > 0)]
        percentile_val = np.percentile(non_empty_sizes, percentile)
                
        return np.where(roi_sizes > percentile_val)[0]

    """
    to_png: save in a directory all slices of a subject
    """
    def to_png(self, sequence, directory):
        img = self.get_sequence(sequence).dataobj        
        for z in range(img.shape[2]):
            imageio.imwrite(f"{directory}/{self.subject}-{sequence}-{z}.png", img[:,:,z])

#    def to_png(self, sequence, z, directory, size=None):
#        img = self.get_sequence(sequence).dataobj       
#            Res=nilearn.image.resample_to_img(BRAIN_BOXES[i], BRAIN_BOXES[def_index],interpolation='nearest')
#        img.resize(size)
        # add resize
#        imageio.imwrite(f"{directory}/{self.subject}-{sequence}-{z}.png", img[:,:,z])

    def bounding_box(self, mask):
        roi = self.get_roi(mask)
        roi_data = roi.dataobj
        
        r = np.any(roi_data, axis=(1, 2))
        c = np.any(roi_data, axis=(0, 2))
        z = np.any(roi_data, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
    
        return ((rmin, rmax), (cmin, cmax), (zmin, zmax))

        """
        delta_r=rmax-rmin
        delta_c=cmax-cmin
        delta_z=zmax-zmin
    
        delta_max=np.amax([delta_r,delta_c,delta_z])

        Rebounded_Sequence=Seq_mask_conv[rmin:(rmin+delta_max),cmin:(cmin+delta_max),zmin:(zmin+delta_max)]

        return Rebounded_Sequence
        """