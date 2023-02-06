import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours, grab_contours
from skimage import measure

import cv2
import skimage
from skimage.filters import threshold_triangle, threshold_sauvola, threshold_otsu
from skimage.filters.rank import median
from skimage. morphology import erosion, disk, closing
from scipy.ndimage.morphology import binary_fill_holes
from scipy import stats
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial import cKDTree
import math
import os
from collections import OrderedDict
from nd2reader import ND2Reader
import matplotlib
matplotlib.rcParams["figure.dpi"] = 300

# set up data directory
folder_names = ['/content/drive/My Drive/patient sample/01272023_p868_D3/Plate001.nd2']
# settings for the filtering
filter_size = [30] # originall 60
disk_size = [3]
threshold_ratio = [1]
threshold_value = [1500]

def get_ordered_list(points, x, y):
    """
    Get ordered list of points
    :param points: unsorted list of points
    :param x: x coordinate of the center
    :param y: y coordinate of the center
    :return: sorted list of points
    """
    points.sort(key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
    return points

# open nd2 file
treatment = 0
folder_to_pro = folder_names[treatment]

x_fluor = []
x_phase = []

with ND2Reader(folder_names[treatment]) as images:
    images.bundle_axes = 'yx'
    # v is the axis for the FOV change
    images.iter_axes = 'v'
    # get phase channel first
    images.default_coords['c'] = 0
    x_phase = []
    for i in range(100):
        a = images[i]
        x_phase.append(a)

    # get fluor channel
    images.default_coords['c'] = 1
    x_fluor = []
    for i in range(100):
        a = images[i]
        x_fluor.append(a)

x_phase = np.stack(x_phase, axis=0)
x_fluor = np.stack(x_fluor, axis=0)

# generate correction phase and fluor images using median filter
avg_p = np.average(x_phase, axis=0)
avg_p1 = avg_p.astype("uint16")
img_pf = median(avg_p1, disk(600))
data_pW = np.divide(x_phase, img_pf)

avg_f = np.average(x_fluor, axis=0)
img_ff = skimage.filters.median(avg_f, disk(filter_size[treatment]))
data_fW = np.divide(x_fluor.astype(np.float64), img_ff)

# plot the images
plt.imshow(img_ff)
plt.figure()
plt.imshow(img_pf)
# print(np.amax(img_ff), np.amin(img_ff))

length = len(data_fW)
# save the cropped images, both corrected and original
crop_totalf = []
crop_totalp = []
crop_totalf_or = []
crop_totalp_or = []

# iterate through each image and crop
for i in range(int(length)):

    mid_slice = data_fW[i, :, :]
    img = skimage.filters.median(mid_slice, disk(disk_size[treatment]))
    thresh_triangle = threshold_triangle(img)
    binary = img > thresh_triangle * threshold_ratio[treatment]
    thresh_erode = erosion(binary)
    thresh_erode_2 = erosion(thresh_erode)

    labels = measure.label(thresh_erode_2, connectivity=2, background=0)
    mask = np.zeros(thresh_erode.shape, dtype="uint8")
    # loop over the unique components
    pixel_count = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
            # otherwise, construct the label mask and count the
            # number of pixels
        labelMask = np.zeros(thresh_erode.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        pixel_count.append(numPixels)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels < threshold_value[treatment] and numPixels > 40:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    if not cnts:
        continue
    cnts = contours.sort_contours(cnts)[0]
    centers = np.zeros((len(cnts), 2))
    # loop over the contours
    # img_p = data_p[90, :, :]
    # img_p8 = cv2.normalize(src=img_p, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    for (r, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        centers[r, :] = [int(cX), int(cY)]
        # cv2.circle(img_p8, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
        # show the output image
    # plt.figure()
    # plt.imshow(img_p8)
    # cells are completely in the image
    centers = centers[centers[:, 1] >= 30]
    centers = centers[centers[:, 1] <= 1009]
    centers = centers[centers[:, 0] >= 30]
    centers = centers[centers[:, 0] <= 1361]

    if not centers.shape[0]:
        continue

    crop_f = []
    crop_p = []
    crop_f_or = []
    crop_p_or = []

    in_stack_p = data_pW[i, :, :]
    in_stack_f = data_fW[i, :, :]

    # iterate over all the centers and crop the image
    for m in range(len(centers)):
        crop = in_stack_p[int(centers[m, 1]) - 30: int(centers[m, 1]) + 30,
               int(centers[m, 0]) - 30: int(centers[m, 0]) + 30]
        # sharpness = cv2.Laplacian(crop, cv2.CV_64F).var()
        # idx = np.argmax(sharpness)
        # best_crop = crop [idx, :, :]
        best_crop_or = x_phase[i, :, :][int(centers[m, 1]) - 30: int(centers[m, 1]) + 30,
                       int(centers[m, 0]) - 30: int(centers[m, 0]) + 30]
        # crop ros image and compute mean intensity

        crop_ros = in_stack_f[int(centers[m, 1]) - 30: int(centers[m, 1]) + 30,
                   int(centers[m, 0]) - 30: int(centers[m, 0]) + 30]
        crop_ros_or = x_fluor[i, :, :][int(centers[m, 1]) - 30: int(centers[m, 1]) + 30,
                      int(centers[m, 0]) - 30: int(centers[m, 0]) + 30]

        crop_p.append(crop)
        crop_f.append(crop_ros)
        crop_p_or.append(best_crop_or)
        crop_f_or.append(crop_ros_or)

    crop_p = np.stack(crop_p, axis=0)
    crop_f = np.stack(crop_f, axis=0)
    crop_p_or = np.stack(crop_p_or, axis=0)
    crop_f_or = np.stack(crop_f_or, axis=0)

    crop_totalp.append(crop_p)
    crop_totalf.append(crop_f)
    crop_totalp_or.append(crop_p_or)
    crop_totalf_or.append(crop_f_or)

crop_totalp = np.concatenate(crop_totalp, axis=0)
crop_totalf = np.concatenate(crop_totalf, axis=0)
crop_totalp_or = np.concatenate(crop_totalp_or, axis=0)
crop_totalf_or = np.concatenate(crop_totalf_or, axis=0)

imgcount = crop_totalf.shape[0]
# if removed, save the reason for removal
skipped = []
skipped_info = []
# save phase channel and ros channel infomation for each cropped image
ros_mean_t = []
ros_back_mean_t = []
phase_mean_t = []
phase_back_mean_t = []

ros_mean_or_t = []
ros_back_mean_or_t = []
phase_mean_or_t = []
phase_back_mean_or_t = []

ros_mean_n = []
ros_back_mean_n = []
phase_mean_n = []
phase_back_mean_n = []

ros_mean_or_n = []
ros_back_mean_or_n = []
phase_mean_or_n = []
phase_back_mean_or_n = []

labels_t = []
coords_num = []
update_coords_num = []
mask_t = []
area_t = []
circularity = []
empty_t = []

# perform segmentation using watershed algorithm and compute mean intensity
for index in range(imgcount):
    phase_slice = crop_totalp[index, :, :]
    gradient_16_th = threshold_sauvola(phase_slice, window_size=25, k=0.01)
    gradient_16_bn = phase_slice > gradient_16_th * 1
    closed = closing(gradient_16_bn, disk(3))
    gradient_16_fill = binary_fill_holes(closed.astype(int))
    distance = ndi.distance_transform_edt(gradient_16_fill)
    coords = peak_local_max(distance, footprint=np.ones((6, 6)), labels=gradient_16_fill)
    coords_num.append(coords.shape[0])
    # remove close points
    # mynumbers = [tuple(point) for point in coords]
    if coords.shape[0] == 2:
        dis = math.sqrt((coords[0, 0] - coords[1, 0]) ** 2 + (coords[0, 1] - coords[1, 1]) ** 2)
        ordered_coords = get_ordered_list(coords.tolist(), 30, 30)
        if dis < 12:
            coords = np.asarray(ordered_coords)[0]
    elif coords.shape[0] > 2:
        ordered_coords = get_ordered_list(coords.tolist(), 30, 30)
        tree = cKDTree(ordered_coords)  # build k-dimensional trie
        pairs = tree.query_pairs(12)  # find all pairs closer than radius: r
        neighbors = dict()
        for i, j in pairs:
            if i in neighbors:
                # append the new number to the existing array at this slot
                neighbors[i].append(j)
            else:
                # create a new array in this slot
                neighbors[i] = [j]
            if j in neighbors:
                # append the new number to the existing array at this slot
                neighbors[j].append(i)
            else:
                # create a new array in this slot
                neighbors[j] = [i]

        sorted_nb = OrderedDict(sorted(neighbors.items()))
        keep = []
        discard = []  # a list would work, but I use a set for fast member testing with `in`
        for iso in range(coords.shape[0]):
            if iso not in sorted_nb:
                keep.append(iso)
        for key in sorted_nb:
            if key not in discard:
                keep.append(key)
                discard.extend(neighbors[key])
        # keep_idx = [x - 1 for x in keep]
        coords = np.asarray(ordered_coords)[keep]

    update_coords_num.append(coords.shape[0])
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=gradient_16_fill)

    labels_t.append(labels)

    # if coords.shape[0] >=5:
    # skipped.append(index)
    # continue

    # labels = watershed(-distance, markers, mask=gradient_16_fill)
    # plt.imshow(labels)
    # save pixel count for each label
    pixel_count = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
            # otherwise, construct the label mask and count the
            # number of pixels
        labelMask = np.zeros(labels.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        pixel_count.append(numPixels)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        # if numPixels < 1500 and numPixels > 80 :
    if not pixel_count:
        empty_t.append(index)
        continue

    m = max(pixel_count)

    a = [i for i, j in enumerate(pixel_count) if j == m]
    mask = np.zeros(labels.shape, dtype="uint8")
    mask[labels == a[0] + 1] = 255
    mask_back = np.zeros(labels.shape, dtype="uint8")
    mask_back[labels == 0] = 255
    mask_back = erosion(erosion(erosion(mask_back)))
    # mask = cv2.add(mask, labelMask)
    cnts, hierachy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = cv2.arcLength(cnts[0], True)
    area = cv2.contourArea(cnts[0])
    if perimeter == 0:
        ratio = 1.1
    else:
        ratio = 4 * math.pi * area / (perimeter ** 2)

    area_t.append(m)
    mask_bn = np.zeros(labels.shape, dtype="uint8")
    mask_bn[labels == a[0] + 1] = 1
    mask_t.append(mask_bn)
    circularity.append(ratio)
    # print(ratio)
    # print(area)
    # hull = cv2.convexHull(cnts[0],returnPoints = False)
    # defects = cv2.convexityDefects(cnts[0],hull)
    # print(defects.shape[0])
    fluor_slice = crop_totalf[index, :, :]

    phase_slice_or = crop_totalp_or[index, :, :]
    fluor_slice_or = crop_totalf_or[index, :, :]

    ros_mean = cv2.mean(fluor_slice, mask)
    ros_back_mean = cv2.mean(fluor_slice, mask_back)
    phase_mean = cv2.mean(phase_slice, mask)
    phase_back_mean = cv2.mean(phase_slice, mask_back)

    # remove a few outliers
    if ros_back_mean[0] == 0:
        skipped.append(index)
        skipped_info.append(sum(pixel_count))

    elif m < 600 or m > 1600:
        skipped.append(index)
        skipped_info.append(m)

    elif ratio < 0.3:
        skipped.append(index)
        skipped_info.append(ratio)

    elif sum(pixel_count) < 600 or sum(pixel_count) > 1600:
        skipped.append(index)
        skipped_info.append(sum(pixel_count))

    elif ros_mean[0] / ros_back_mean[0] < 1:
        skipped.append(index)
        skipped_info.append(ros_mean[0] / ros_back_mean[0])

    ros_mean_or = cv2.mean(fluor_slice_or, mask)
    ros_back_mean_or = cv2.mean(fluor_slice_or, mask_back)
    phase_mean_or = cv2.mean(phase_slice_or, mask)
    phase_back_mean_or = cv2.mean(phase_slice_or, mask_back)

    ros_mean_t.append(ros_mean[0])
    ros_back_mean_t.append(ros_back_mean[0])
    phase_mean_t.append(phase_mean[0])
    phase_back_mean_t.append(phase_back_mean[0])

    ros_mean_or_t.append(ros_mean_or[0])
    ros_back_mean_or_t.append(ros_back_mean_or[0])
    phase_mean_or_t.append(phase_mean_or[0])
    phase_back_mean_or_t.append(phase_back_mean_or[0])

    #

ros_mean_t = np.stack(ros_mean_t, axis=0)
ros_back_mean_t = np.stack(ros_back_mean_t, axis=0)
phase_mean_t = np.stack(phase_mean_t, axis=0)
phase_back_mean_t = np.stack(phase_back_mean_t, axis=0)

ros_mean_or_t = np.stack(ros_mean_or_t, axis=0)
ros_back_mean_or_t = np.stack(ros_back_mean_or_t, axis=0)
phase_mean_or_t = np.stack(phase_mean_or_t, axis=0)
phase_back_mean_or_t = np.stack(phase_back_mean_or_t, axis=0)

labels_t = np.stack(labels_t, axis=0)
coords_num = np.stack(coords_num, axis=0)
update_coords_num = np.stack(update_coords_num, axis=0)
mask_t = np.stack(mask_t, axis=0)
area_t = np.stack(area_t, axis=0)
circularity = np.stack(circularity, axis=0)

total_list = list(range(imgcount))
list_remain = [x for x in total_list if x not in empty_t]  # %% use this to extract all images
raw_dataidx = [x for x in list_remain if x not in skipped]
sort_skipped = [list_remain.index(skipped[i]) for i in range(len(skipped))]
sort_dataidx = [x for x in list(range(len(list_remain))) if x not in sort_skipped]

plt.plot(ros_mean_t)
plt.plot(ros_back_mean_t)
plt.figure()
ros_r = np.divide(ros_mean_t, ros_back_mean_t)[sort_dataidx]
plt.plot(ros_r)