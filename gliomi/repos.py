import nibabel as nb
import os

class SequenceRepoPasquini:
    
    def __init__(self, subjects_dir):
        self.subjects_dir = subjects_dir
        
    def get_sequence(self, subject, sequence_name):
        return nb.load(f"{self.subjects_dir}/{subject}/{sequence_name}_registered.nii.gz")

    def get_roi(self, subject, mask_name="T2ROI"):
        return nb.load(f"{self.subjects_dir}/{subject}/ROI/{mask_name}.nii.gz")
    
    def has(self, subject, sequence_name):
        return os.path.exists(f"{self.subjects_dir}/{subject}/{sequence_name}_registered.nii.gz")

class SequenceRepoGliomi:
    
    def __init__(self, subjects_dir):
        self.subjects_dir = subjects_dir
        
    def get_sequence(self, subject, sequence_name):
        return nb.load(f"{self.subjects_dir}/scaled/{subject}/{sequence_name}scalatamedia.nii")

    def get_roi(self, subject, mask_name="T2ROI"):
        return nb.load(f"{self.subjects_dir}/registered/{subject}/ROI/{mask_name}.nii")    
    
    def has(self, subject, sequence_name):
        return os.path.exists(f"{self.subjects_dir}/scaled/{subject}/{sequence_name}scalatamedia.nii")
    
class SequenceRepoBrats19:
    
    def __init__(self, subjects_dir):
        self.subjects_dir = subjects_dir
        
    def get_sequence(self, subject, sequence_name):
        return nb.load(f"{self.subjects_dir}/scaled/{subject}/{subject}_{sequence_name}_scaled.nii")

    def get_roi(self, subject):
        return nb.load(f"{self.subjects_dir}/HGG_Training_data_Brats_2019/{subject}/{subject}_seg.nii.gz") 

    def has(self, subject, sequence_name):
        return os.path.exists(f"{self.subjects_dir}/HGG_Training_data_Brats_2019/{subject}/{subject}_{sequence_name}.nii.gz")
