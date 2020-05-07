import nibabel as nb
import numpy as np
import imageio
from nilearn.plotting import plot_anat
from PIL import Image
import nilearn

"""
returns the bounding box of a mask (.nii file)
"""
def get_bounding_box(mask):

    roi = mask.dataobj

    r = np.any(roi, axis=(1, 2))
    c = np.any(roi, axis=(0, 2))
    z = np.any(roi, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return ((rmin, rmax), (cmin, cmax), (zmin, zmax))

"""
Get the part of sequence masked and related to its bounding box
"""
def mask_and_crop(sequence, mask):

    ((rmin, rmax), (cmin, cmax), (zmin, zmax)) = get_bounding_box(mask)

    delta_r = rmax-rmin
    delta_c = cmax-cmin
    delta_z = zmax-zmin

    data = sequence.get_fdata()[rmin:(rmin+delta_r),cmin:(cmin+delta_c),zmin:(zmin+delta_z)]

    return nb.Nifti1Image(
        data, 
        affine=sequence.affine)

"""
Extract mask from a sequence and resize to a cube a a given side
"""
def mask_crop_resize(sequence, mask, side):

    roi = mask_and_crop(sequence, mask)

    (dim1, dim2, dim3) = float(roi.shape[0]), float(roi.shape[1]), float(roi.shape[2])
    
    scale_affine = np.array([[float(side) / dim1, 0, 0, 0], 
                             [0, float(side) / dim2, 0, 0], 
                             [0, 0, float(side) / dim3, 0], 
                             [0, 0, 0, 1]])

    resampled_roi = nb.Nifti1Image(
        roi.dataobj, 
        affine=scale_affine)

    return nilearn.image.resample_img(
        resampled_roi, 
        target_affine=np.eye(4),
        target_shape=(side, side, side), 
        interpolation='nearest')

"""
Subject data
"""
class Subject():
    
    def __init__(self, subject_dir, subject):
        self.subject_dir = subject_dir
        self.subject = subject
        self.sequence = {}
        self.roi = None

    """
    get_sequence: returns a sequence given a file for a sequence (e.g. T1, T2, ADC...)
    """
    def get_sequence(self, sequence):
        if sequence in self.sequence:
            return self.sequence['sequence']
        self.sequence['sequence'] = nb.load(f"{self.subject_dir}/{self.subject}/{sequence}_registered.nii")
        return self.sequence['sequence']
    
    """
    get_roi: returns a roi given a a roi type (e.g. SOLID)
    """
    def get_roi(self, roi_type):
        if self.roi:
            return self.roi
        self.roi = nb.load(f"{self.subject_dir}/{self.subject}/ROI/{roi_type}.nii")
        return self.roi

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
    Plot a sequence for a subject
    """
    def plot(self, sequence="T1", display_mode="ortho"):
        plot_anat(self.get_sequence(sequence), cmap='magma', colorbar=False, display_mode=display_mode, annotate=False);

    """
    to_png: save in a directory all slices of a subject according Z axis
    """
    def to_png(self, sequence, directory):
        img = self.get_sequence(sequence).dataobj        
        for z in range(img.shape[2]):
            imageio.imwrite(f"{directory}/{self.subject}-{sequence}-{z}.png", img[:,:,z])
    
    """
    def percentile_slicer(sequence_mask, brain_box, percentile,affine):
        length=len(sequence_mask)
        for i in range(length):
            seq_sizes=np.sum(sequence_mask,axis=(0,1))
            non_empty_sizes = seq_sizes[np.where(seq_sizes > 0)]
            percentile_val = np.percentile(non_empty_sizes, percentile)
            seq_indexes = np.where(seq_sizes > percentile_val)[0]
    
        return np.asarray([brain_box[:,:,k] for k in seq_indexes])
    
    def rebound_with_mask(Seq_mask_conv):
        r = np.any(Seq_mask_conv, axis=(1, 2))
        c = np.any(Seq_mask_conv, axis=(0, 2))
        z = np.any(Seq_mask_conv, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
    
    #print(rmin,rmax,cmin,cmax,zmin,zmax)
    
        delta_r=rmax-rmin
        delta_c=cmax-cmin
        delta_z=zmax-zmin
    
        delta_max=np.amax([delta_r,delta_c,delta_z])
    
        Rebounded_Sequence=Seq_mask_conv[rmin:(rmin+delta_max),cmin:(cmin+delta_max),zmin:(zmin+delta_max)]
    
        return Rebounded_Sequence
    """
  