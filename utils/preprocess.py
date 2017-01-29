#based on https://www.kaggle.com/c/data-science-bowl-2017/details/tutorial and https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
import numpy as np
import sys
import os
#import pandas as pd
import scipy.ndimage.interpolation as interp

import dicom

from glob import glob



class ct_scan:
    def __init__(self, scan_token, base_dir = "./sample_images/"):
        self.base_dir = "./sample_images/"
        self.scan_token = scan_token
        self.set_resampling_spacing([1.,1.,1.])

    def set_resampling_spacing(self, new_spacings):
        self.resampling_spacing=new_spacings

    # get filenames of all slices for one scan/patient
    def get_flist_one_scan(self):
        f_list = glob(self.base_dir + self.scan_token + "/*.dcm")
        return f_list

    # get slice thickness
    def get_slice_thickness(self):
        if not hasattr(self, 'spacings'):
            self.get_slice_spacings()
        return self.spacings[0]

    # get slice spacings
    def get_slice_spacings(self):
        f_list_ = self.get_flist_one_scan()
        dicoms_ = [dicom.read_file(f_) for f_ in f_list_]
        dicoms_.sort(key=lambda x: int(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(dicoms_[0].ImagePositionPatient[2] - dicoms_[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(dicoms_[0].SliceLocation - dicoms_[1].SliceLocation)

        pixel_spacingsx = float(dicoms_[0].PixelSpacing[0])
        pixel_spacingsy = float(dicoms_[0].PixelSpacing[1])

        #self.slice_thickness = slice_thickness
        #self.pixel_spacingsx = pixel_spacingsx
        #self.pixel_spacingsy = pixel_spacingsy
        self.spacings = np.array([slice_thickness, pixel_spacingsx, pixel_spacingsy])
        #self.spacings = spacings
        #return slice_thickness, pixel_spacingsx, pixel_spacingsy

    # get data of all slices for one scan/patient
    def get_slices_one_scan(self):
        f_list_ = self.get_flist_one_scan()
        dicoms_ = [dicom.read_file(f_) for f_ in f_list_]
        dicoms_.sort(key=lambda x: int(x.ImagePositionPatient[2]))

        num_slices = len(f_list_)
        # we verified that all slices are 512x512
        slices = np.zeros((num_slices, 512, 512)).astype('float')

        for i, slice_ in enumerate(dicoms_):
            # ds_ = dicom.read_file(slice_f_)
            slices[i] = slice_.pixel_array
        # replace non-measurements with 0s
        slices[slices == -2000] = 0
        self.slices = slices
        #return slices

    # resampling so that the scales are consistent
    def resample(self, slices=None, new_spacings=None):
        if slices is None:
            if not hasattr(self, 'slices'):
                self.get_slices_one_scan()
            slices = self.slices
        if new_spacings is None:
            new_spacings = self.resampling_spacing
        # Determine current pixel spacing
        #slice_thickness, pixel_spacingsx, pixel_spacingsy = self.get_slice_spacings()
        #spacings = np.array([slice_thickness, pixel_spacingsx, pixel_spacingsy])
        self.get_slice_spacings()

        resize_factor = self.spacings / new_spacings
        new_real_shape = slices.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / slices.shape
        new_spacings = self.spacings / real_resize_factor

        resampled_slices = interp.zoom(slices, real_resize_factor, mode='nearest')
        self.new_spacings = new_spacings
        self.resampled_slices = resampled_slices
        #return slices, new_spacings


def get_images(scan_token, base_dir = "./sample_images/", resample=True):
    ct = ct_scan(scan_token, base_dir=base_dir)
    if resample:
        ct.resample()
        return ct.resampled_slices, ct.new_spacings
    else:
        ct.get_slices_one_scan()
        ct.get_slice_spacings()
        return ct.slices, ct.spacings


# get ids for all scans/patients
def get_scan_ids(base_dir = "./sample_images/"):
    scans_ = []
    for scan_ in os.listdir(base_dir):
        if not scan_.startswith('.'):
            scans_.append(scan_)
    return scans_


def get_all_images(base_dir = "./sample_images/", resample=True):
    all_scan_ids = get_scan_ids(base_dir)
    for scan_id_ in all_scan_ids:
        slices_, spacings_ = get_images(scan_id_, base_dir=base_dir, resample=resample)
