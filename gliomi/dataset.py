import nibabel as nb
import numpy as np
import nilearn
import os
import pickle
from nilearn.plotting import plot_anat

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
def mask_and_crop(sequence, mask, full_brain=False):
    
    assert sequence.shape == mask.shape

    ((rmin, rmax), (cmin, cmax), (zmin, zmax)) = get_bounding_box(mask)
    
    if full_brain:
        rmin, rmax = 0, mask.shape[0]
        cmin, cmax = 0, mask.shape[1]

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
def mask_crop_resize(sequence, mask, x, y, z, full_brain=False):

    roi = mask_and_crop(sequence, mask, full_brain)

    (dim1, dim2, dim3) = float(roi.shape[0]), float(roi.shape[1]), float(roi.shape[2])
    
    scale_affine = np.array([[float(x) / dim1, 0, 0, 0], 
                             [0, float(y) / dim2, 0, 0], 
                             [0, 0, float(z) / dim3, 0], 
                             [0, 0, 0, 1]])

    resampled_roi = nb.Nifti1Image(
        roi.dataobj, 
        affine=scale_affine)

    return nilearn.image.resample_img(
        resampled_roi, 
        target_affine=np.eye(4),
        target_shape=(x, y, z), 
        interpolation='nearest')

"""
Establish ROI size around an axis (z-axis is 2)
"""
def get_roi_size(roi, axis):
    
    roi_data = roi.get_fdata()

    # This is required by Brats since ROI is 0, 1, 2, 4 instead only 0, 1
    roi_data[roi_data>1] = 1

    other_axis = list(range(len(roi_data.shape)))
    
    other_axis.remove(axis)

    return np.sum(roi_data, axis=tuple(other_axis))

"""
Get index of image according and axis given the size of ROI
along that axis (z-index is 2)
"""
def get_roi_index_percentile(roi, axis, percentile):
    
    # Sizes along z-axis
    roi_sizes = get_roi_size(roi, axis)
        
    non_empty_sizes = roi_sizes[np.where(roi_sizes > 0)]
    
    percentile_val = np.percentile(non_empty_sizes, percentile)

    return np.where(roi_sizes >= percentile_val)[0]

def index_percentile_of_sizes(sizes, percentile):
    
    non_empty_sizes = sizes[np.where(sizes > 0)]
    
    percentile_val = np.percentile(non_empty_sizes, percentile)

    return np.where(sizes >= percentile_val)

"""
Return ordered index of sizes index with a given percentile
"""
def ordered_index_percentile_of_sizes(sizes, percentile):
    
    non_empty_sizes = sizes[np.where(sizes > 0)]
    
    percentile_val = np.percentile(non_empty_sizes, percentile)

    w = np.where(sizes >= percentile_val)

    sort_index = np.argsort(sizes)

    r = sort_index[-w[0].shape[0]:]
    
    return r[::-1]

"""
Save the tumor crop with base shaped in square with given side.
"""
def get_slices_for_subject(sequence_repo, sequence_name, subject, side, full_brain=False):
        
    sequence = sequence_repo.get_sequence(subject, sequence_name)
    
    roi = sequence_repo.get_roi(subject)

    ((rmin, rmax), (cmin, cmax), (zmin, zmax)) = get_bounding_box(roi)
    
    z_height = zmax - zmin
    
    sequence_resampled = mask_crop_resize(sequence, roi, side, side, z_height, full_brain)
    
    slices = sequence_resampled.get_fdata()
    
    return slices
        
"""
Save the tumor crop with base shaped in square with given side.
"""
"""
def save_slices_for_subject(sequence_repo, sequence_name, subject, side, output_dir, full_brain=False):
    
    slices = get_slices_for_subject(sequence_repo, sequence_name, subject, side, full_brain)
                
    with open(f"{output_dir}/{subject}/slices-{sequence_name}-{side}.pickle", "wb") as out:
        pickle.dump(slices, out)
"""

"""
Save the tumor crop with base shaped in square with given side.
"""
"""
def save_cube_for_subject(sequence_repo, sequence_name, subject, side, output_dir):
        
    sequence = sequence_repo.get_sequence(subject, sequence_name)
    
    roi = sequence_repo.get_roi(subject, "T2ROI")

    resampled_roi = mask_crop_resize(roi, roi, side, side, side)
    
    sequence_resampled = mask_crop_resize(sequence, roi, side, side, side)
    
    slices = sequence_resampled.get_fdata()
        
    with open(f"{output_dir}/{subject}/cube-{sequence_name}-{side}.pickle", "wb") as out:
        pickle.dump(slices, out)
"""
        
"""
Save the slices of the whole brain reshaped with a squared size
"""
"""
def save_slices_for_subject_full_brain(sequence_repo, sequence_name, subject, side, output_dir):
        
    sequence = sequence_repo.get_sequence(subject, sequence_name)
    
    roi = sequence_repo.get_roi(subject, "T2ROI")

    ((rmin, rmax), (cmin, cmax), (zmin, zmax)) = get_bounding_box(roi)
    
    z_height = zmax - zmin

    sequence_resampled = mask_full_brain_resize(sequence, roi, side, side, z_height)
    
    slices = sequence_resampled.get_fdata()
        
    with open(f"{output_dir}/{subject}/slices-{sequence_name}-{side}.pickle", "wb") as out:
        pickle.dump(slices, out)
"""

"""
This method is able to normalize (like standard scaler) but with the possibility to specify axis
"""
def normalize(images, max_value, axis):
    
    u, s = np.mean(images, axis=axis), np.std(images, axis=axis)
    
    u_extended = np.expand_dims(u, axis=axis)
    s_extended = np.expand_dims(s, axis=axis)
    
    images_centered = (images - u_extended) / s_extended
    
    max_ = np.max(images_centered, axis=axis)
    min_ = np.min(images_centered, axis=axis)
    max_extended = np.expand_dims(max_, axis=axis)
    min_extended = np.expand_dims(min_, axis=axis)
    
    delta_ = max_extended - min_extended
    
    return ((images_centered - min_extended) / delta_) * max_value

"""
def save_slices_for_subject_brats19(sequence_repo, sequence_name, subject, side, output_dir):
        
    sequence = sequence_repo.get_sequence(subject, sequence_name)
    
    roi = sequence_repo.get_roi(subject)

    ((rmin, rmax), (cmin, cmax), (zmin, zmax)) = get_bounding_box(roi)
    
    z_height = zmax - zmin
    
    sequence_resampled = mask_crop_resize(sequence, roi, side, side, z_height)
    
    slices = sequence_resampled.get_fdata()
        
    with open(f"{output_dir}/{subject}/slices-{sequence_name}-{side}.pickle", "wb") as out:
        pickle.dump(slices, out)
"""     
            
def save_slices_for_subject_full_brain_brats19(sequence_repo, sequence_name, subject, side, output_dir):
        
    sequence = sequence_repo.get_sequence(subject, sequence_name)
    
    roi = sequence_repo.get_roi(subject)

    ((rmin, rmax), (cmin, cmax), (zmin, zmax)) = get_bounding_box(roi)
    
    z_height = zmax - zmin

    sequence_resampled = mask_full_brain_resize(sequence, roi, side, side, z_height)
    
    slices = sequence_resampled.get_fdata()
        
    with open(f"{output_dir}/{subject}/slices-{sequence_name}-{side}.pickle", "wb") as out:
        pickle.dump(slices, out)
