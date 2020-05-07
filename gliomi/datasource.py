"""
Make possible to generate datasource based on MRI data
"""

import os
import numpy as np
import imageio
import sys
from .subject import *

from nilearn.image import resample_to_img

class DatasourceGenerator():
    
    def __init__(self, subject_dir, template_subject, skip_subjects):
        self.subject_dir = subject_dir
        self.template_subject = template_subject
        self.skip_subjects = skip_subjects
    
    """
    Generate a dataset:
    Each subject will held its slices in a directory with its name
    """
    def masked_by_percentile(self, dataset_path, sequence, mask="T2ROI", percentile=70):
    
        print(f"Generating dataset for {sequence} with percentile {percentile}.")
        print(f"We are using {self.template_subject} as template.")
        print(f"We are skipping {self.skip_subjects}")

        # sequence for template subject
        orig = Subject(self.subject_dir, self.template_subject)
        t1_orig = orig.get_sequence(sequence)

        subjects = [x[0] for x in os.walk(self.subject_dir)]

        for subject in subjects:
            
            if subject in self.skip_subjects:
                next
    
            print(f"Try fetch slices for {subject}")

            try:

                s = Subject(datasetDir, subject)

                dst = f"{dataset_path}/{subject}"

                os.makedirs(dst, exist_ok=True)

                t1 = s.get_sequence(sequence)
                t1_reshaped = resample_to_img(t1, t1_orig, interpolation='nearest')

                roi = s.get_roi(mask)
                roi_shaped = resample_to_img(roi, t1_orig, interpolation='nearest')

                roi_shaped_data = roi_shaped.get_fdata()
                roi_sizes = np.sum(roi_shaped_data, axis=(0, 1))
                non_empty_sizes = roi_sizes[np.where(roi_sizes > 0)]
                percentile_val = np.percentile(non_empty_sizes, percentile)
                roi_indexes = np.where(roi_sizes > percentile_val)[0]

                # Getting resampled subject data
                # img = t1_reshaped.dataobj

                img = np.asarray(t1_reshaped.dataobj)
                data_min = np.min(img)
                data_max = np.max(img)
                data_delta = data_max - data_min
                img = np.uint8((img - data_min) / data_delta * 255)

                # Save PNG for each slice with mask size above the selected percentile
                for z in roi_indexes:
                    imageio.imwrite(f"{dst}/{z}.png", img[:,:,z])

            except FileNotFoundError as e:
                print(e)
            except TypeError as e:
                print(e)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                print(f"Skipping subject {subject}: missing sequence {sequence}")
    
    