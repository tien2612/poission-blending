import tkinter
from tkinter import filedialog
from PIL import Image
import numpy as np
from scipy.sparse import csr_matrix
from pyamg.gallery import poisson
from pyamg import ruge_stuben_solver
import matplotlib.pyplot as plt
from skimage.draw import polygon


def get_image_path_from_user(msg):
    tkinter.Tk().withdraw()
    return tkinter.filedialog.askopenfilename(title=msg)


def rgb_to_gray_mat(img_path):
    gray_img = Image.open(img_path).convert('L')
    return np.asarray(gray_img)


def get_image_from_user(msg, src_shape=(0, 0)):
    img_path = get_image_path_from_user(msg)
    rgb = split_image_to_rgb(img_path)
    if not np.all(np.asarray(src_shape) < np.asarray(rgb[0].shape)):
        return get_image_from_user('Open destination image with resolution bigger than ' +
                                   str(tuple(np.asarray(src_shape) + 1)), src_shape)
    return img_path, rgb


def poly_mask(img_path, num_of_pts=100):
    img = rgb_to_gray_mat(img_path)
    plt.figure('source image')
    plt.title('Inscribe the region you would like to blend inside a polygon')
    plt.imshow(img, cmap='gray')
    pts = np.asarray(plt.ginput(num_of_pts, timeout=-1))
    plt.close('all')
    if len(pts) < 3:
        min_row, min_col = (0, 0)
        max_row, max_col = img.shape
        mask = np.ones(img.shape)
    else:
        pts = np.fliplr(pts)
        in_poly_row, in_poly_col = polygon(tuple(pts[:, 0]), tuple(pts[:, 1]), img.shape)
        min_row, min_col = (np.max(np.vstack([np.floor(np.min(pts, axis=0)).astype(int).reshape((1, 2)), (0, 0)]),
                             axis=0))
        max_row, max_col = (np.min(np.vstack([np.ceil(np.max(pts, axis=0)).astype(int).reshape((1, 2)), img.shape]),
                             axis=0))
        mask = np.zeros(img.shape)
        mask[in_poly_row, in_poly_col] = 1
        mask = mask[min_row: max_row, min_col: max_col]
    return mask, min_row, max_row, min_col, max_col


def split_image_to_rgb(img_path):
    r, g, b = Image.Image.split(Image.open(img_path))
    return np.asarray(r), np.asarray(g), np.asarray(b)


def crop_image_by_limits(src, min_row, max_row, min_col, max_col):
    r, g, b = src
    r = r[min_row: max_row, min_col: max_col]
    g = g[min_row: max_row, min_col: max_col]
    b = b[min_row: max_row, min_col: max_col]
    return r, g, b


def keep_src_in_dst_boundaries(corner, gray_dst_shape, src_shape):
    for idx in range(len(corner)):
        if corner[idx] < 1:
            corner[idx] = 1
        if corner[idx] > gray_dst_shape[idx] - src_shape[idx] - 1:
            corner[idx] = gray_dst_shape[idx] - src_shape[idx] - 1
    return corner


def top_left_corner_of_src_on_dst(dst_img_path, src_shape):
    gray_dst = rgb_to_gray_mat(dst_img_path)
    plt.figure('destination image')
    plt.title('Where would you like to blend it..?')
    plt.imshow(gray_dst, cmap='gray')
    center = np.asarray(plt.ginput(2, -1, True)).astype(int)
    plt.close('all')
    if len(center) < 1:
        center = np.asarray([[gray_dst.shape[1] // 2, gray_dst.shape[0] // 2]]).astype(int)
    elif len(center) > 1:
        center = np.asarray([center[0]])
    corner = [center[0][1] - src_shape[0] // 2, center[0][0] - src_shape[1] // 2]
    return keep_src_in_dst_boundaries(corner, gray_dst.shape, src_shape)


def crop_dst_under_src(dst_img, corner, src_shape):
    dst_under_src = dst_img[
                  corner[0]:corner[0] + src_shape[0],
                  corner[1]:corner[1] + src_shape[1]]
    return dst_under_src


def laplacian(array):
    return poisson(array.shape, format='csr') * csr_matrix(array.flatten()).transpose().toarray()


def set_boundary_condition(b, dst_under_src):
    b[1, :] = dst_under_src[1, :]
    b[-2, :] = dst_under_src[-2, :]
    b[:, 1] = dst_under_src[:, 1]
    b[:, -2] = dst_under_src[:, -2]
    b = b[1:-1, 1: -1]
    return b


def construct_const_vector(mask, mixed_grad, dst_under_src, src_laplacianed, src_shape):
    dst_laplacianed = laplacian(dst_under_src)
    b = np.reshape((1 - mixed_grad) * mask * np.reshape(src_laplacianed, src_shape) +
                   mixed_grad * mask * np.reshape(dst_laplacianed, src_shape) +
                   (mask - 1) * (- 1) * np.reshape(dst_laplacianed, src_shape), src_shape)
    return set_boundary_condition(b, dst_under_src)


def fix_coeff_under_boundary_condition(coeff, shape):
    shape_prod = np.prod(np.asarray(shape))
    arange_space = np.arange(shape_prod).reshape(shape)
    arange_space[1:-1, 1:-1] = -1
    index_to_change = arange_space[arange_space > -1]
    for j in index_to_change:
        coeff[j, j] = 1
        if j - 1 > -1:
            coeff[j, j - 1] = 0
        if j + 1 < shape_prod:
            coeff[j, j + 1] = 0
        if j - shape[-1] > - 1:
            coeff[j, j - shape[-1]] = 0
        if j + shape[-1] < shape_prod:
            coeff[j, j + shape[-1]] = 0
    return coeff


def construct_coefficient_mat(shape):
    a = poisson(shape, format='lil')
    a = fix_coeff_under_boundary_condition(a, shape)
    return a


def build_linear_system(mask, src_img, dst_under_src, mixed_grad):
    src_laplacianed = laplacian(src_img)
    b = construct_const_vector(mask, mixed_grad, dst_under_src, src_laplacianed, src_img.shape)
    a = construct_coefficient_mat(b.shape)
    return a, b


def solve_linear_system(a, b, b_shape):
    multi_level = ruge_stuben_solver(csr_matrix(a))
    x = np.reshape((multi_level.solve(b.flatten(), tol=1e-10)), b_shape)
    x[x < 0] = 0
    x[x > 255] = 255
    return x


def blend(dst, patch, corner, patch_shape, blended):
    mixed = dst.copy()
    mixed[corner[0]:corner[0] + patch_shape[0], corner[1]:corner[1] + patch_shape[1]] = patch
    blended.append(Image.fromarray(mixed))
    return blended


def poisson_and_naive_blending(mask, corner, src_rgb, dst_rgb, mixed_grad):
    poisson_blended = []
    naive_blended = []
    for color in range(3):
        src = src_rgb[color]
        dst = dst_rgb[color]
        dst_under_src = crop_dst_under_src(dst, corner, src.shape)
        a, b = build_linear_system(mask, src, dst_under_src, mixed_grad)
        x = solve_linear_system(a, b, b.shape)
        poisson_blended = blend(dst, x, (corner[0] + 1, corner[1] + 1), b.shape, poisson_blended)
        crop_src = mask * src + (mask - 1) * (- 1) * dst_under_src
        naive_blended = blend(dst, crop_src, corner, src.shape, naive_blended)
    return poisson_blended, naive_blended


def merge_save_show(split_img, img_title):
    merged = Image.merge('RGB', tuple(split_img))
    merged.save(img_title + '.png')
    merged.show(img_title)


def main():
    src_img_path, src_rgb = get_image_from_user('Open source image')
    mask, *mask_limits = poly_mask(src_img_path)
    src_rgb_cropped = crop_image_by_limits(src_rgb, *mask_limits)
    dst_img_path, dst_rgb = get_image_from_user('Open destination image', src_rgb_cropped[0].shape)
    corner = top_left_corner_of_src_on_dst(dst_img_path, src_rgb_cropped[0].shape)
    poisson_blended, naive_blended = poisson_and_naive_blending(mask, corner, src_rgb_cropped, dst_rgb, 0.3)
    merge_save_show(naive_blended, 'Naive Blended')
    merge_save_show(poisson_blended, 'Poisson Blended')


if __name__ == '__main__':
    main()
