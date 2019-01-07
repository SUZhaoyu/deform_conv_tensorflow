from __future__ import division

import cv2
import imageio
import numpy as np


def zoom(img, zoom_factor):
    x_length = img.shape[0]
    y_length = img.shape[1]
    dim = len(img.shape)
    back_ground = np.zeros(img.shape, dtype=np.uint8)
    x_width = int(zoom_factor * x_length)
    y_width = int(zoom_factor * y_length)
    img = cv2.resize(img, (y_width, x_width), interpolation=cv2.INTER_CUBIC)
    if x_width < x_length:
        x_start = int((x_length - x_width) / 2)
        y_start = int((y_length - y_width) / 2)
        if dim == 3:
            back_ground[x_start:x_start + x_width, y_start:y_start + y_width, :] = img
        elif dim == 2:
            back_ground[x_start:x_start + x_width, y_start:y_start + y_width] = img
        else:
            raise ValueError("Input dim is incorrect")
        return back_ground
    else:
        x_start = int((x_width - x_length) / 2)
        y_start = int((y_width - y_length) / 2)
        if dim == 3:
            return img[x_start:x_start + x_length, y_start:y_start + y_length, :]
        elif dim == 2:
            return img[x_start:x_start + x_length, y_start:y_start + y_length]
        else:
            raise ValueError("Input dim is incorrect")


def rotate(img, rotate_angle):
    x_length = img.shape[0]
    y_length = img.shape[1]
    M = cv2.getRotationMatrix2D((x_length // 2, y_length // 2), rotate_angle, 1)
    img = cv2.warpAffine(img, M, img.shape[:2])
    return img


def get_scaled_imgs(img, total_frames=60, zoom_factor=[0.3, 1.0]):
    img = zoom(img, zoom_factor=1.2)  # The image is first expanded by 120%
    # img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_NEAREST)
    shrink_factors = np.linspace(zoom_factor[1], zoom_factor[0], num=total_frames // 2)
    expand_factors = shrink_factors[::-1]
    rotate_angles_clock = np.linspace(0, 360, num=total_frames // 2)
    rotate_angles_counter_clock = rotate_angles_clock[::-1]
    frames = []
    for i in range(total_frames // 2):
        new_img = zoom(img, zoom_factor=shrink_factors[i])
        frames.append(rotate(new_img, rotate_angle=rotate_angles_clock[i]))
    for i in range(total_frames // 2):
        new_img = zoom(img, zoom_factor=expand_factors[i])
        frames.append(rotate(new_img, rotate_angle=rotate_angles_clock[i]))

    return np.array(frames)


def get_activation_mask(img, stride, thres=10):
    mask = []
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] >= thres:
                x_, y_ = x // stride, y // stride
                if [x_, y_] not in mask:
                    mask.append([x_, y_])
    return mask


def get_pad_dim(img, offset, stride, kernel_size, padding):
    if padding.upper() == 'SAME':
        in_height = img.shape[0]
        in_width = img.shape[1]
        out_height = offset.shape[-2]
        out_width = offset.shape[-1]
        pad_along_height = ((out_height - 1) * stride + kernel_size - in_height)
        pad_along_width = ((out_width - 1) * stride + kernel_size - in_width)
        pad_top = pad_along_height // 2
        pad_left = pad_along_width // 2
        return pad_top, pad_left
    elif padding.upper() == 'VALID':
        return 0, 0
    else:
        raise ValueError("Unsupported padding mothod: {}".format(padding))


def plot_offset_point(img, offset_coor, scale):
    offset_x = offset_coor[0] * scale
    offset_y = offset_coor[1] * scale
    low_x = int(np.floor(offset_x))
    high_x = int(low_x + 1)
    low_y = int(np.floor(offset_y))
    high_y = int(low_y + 1)
    if low_x * low_y >= 0 and (high_x - img.shape[0]) * (high_y - img.shape[1]) >= 0:
        cv2.rectangle(img, (low_y, low_x), (high_y, high_x), (255, 0, 0), 2)
    return img


def plot_offset(img, offset, kernel_size, stride, padding, thres=0.5, group_id=0, expand_scale=10, text=None):
    pad_top, pad_left = get_pad_dim(img, offset, stride, kernel_size, padding)
    offset_channels = 2 * kernel_size ** 2
    group_offset = offset[group_id * offset_channels:(group_id + 1) * offset_channels, :, :]
    plot_img = cv2.resize(img, (img.shape[0] * expand_scale, img.shape[1] * expand_scale),
                          interpolation=cv2.INTER_NEAREST)
    plot_img = np.repeat(plot_img[:, :, np.newaxis], 3, axis=2)
    for x in range(offset.shape[-2]):
        for y in range(offset.shape[-1]):
            kernel_offset = group_offset[:, x, y]
            kernel_offset = np.reshape(kernel_offset, (kernel_size, kernel_size, 2))
            input_start_x = x * stride - pad_top
            input_start_y = y * stride - pad_left
            for i in range(kernel_size):
                for j in range(kernel_size):
                    input_x = input_start_x + i
                    input_y = input_start_y + j
                    offset_x = kernel_offset[i, j, 0]
                    offset_y = kernel_offset[i, j, 1]
                    if np.abs(offset_x) >= thres or np.abs(offset_y) >= thres:
                        offset_coor = [input_x + offset_x + 0.5, input_y + offset_y + 0.5]
                        plot_img = plot_offset_point(plot_img, offset_coor, expand_scale)
    if text is not None:
        cv2.putText(plot_img, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1, cv2.CV_AA)
    return plot_img


def plot_concat(imgs, space=0.1):
    img_width = imgs[0].shape[1]
    img_height = imgs[0].shape[0]
    blank = int(space * img_width)
    back_ground = np.ones((img_height, img_width * len(imgs) + blank * (len(imgs) - 1), 3)) * 255
    back_ground = back_ground.astype(np.uint8)
    for i in range(len(imgs)):
        w = i * (img_width + blank)
        back_ground[:, w:w + img_width, :] = imgs[i]
    return back_ground


def plot_gif(frames, gif_name='img/offset_viz.gif'):
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.07)


if __name__ == '__main__':
    img = cv2.imread('img.png')[:, :, 0].astype(np.uint8)
    offset = np.zeros((18, 28, 28))
    frame = plot_offset(img=img,
                        offset=offset,
                        kernel_size=3,
                        stride=1,
                        padding="SAME")
    cv2.imwrite('test.png', frame)

# print(np.max(img))
# mask = get_activation_mask(img, stride=1)
# print(mask)
# frames = get_scaled_imgs(img)
# plot_gif(frames)
