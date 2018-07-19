# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


def draw_rect(filename, rect_list):
    raw_img = Image.open(filename)
    draw = ImageDraw.Draw(raw_img)
    x_begin = rect_list[0]
    y_begin = rect_list[1]
    x_end = rect_list[0] + rect_list[2]
    y_end = rect_list[1] + rect_list[3]
    region = (x_begin, y_begin, x_end, y_end)
    croped_img = raw_img.crop(region)
    draw.line([(x_begin, y_begin), (x_end, y_begin)], fill=(255, 0, 0), width=5)
    draw.line([(x_end, y_begin), (x_end, y_end)], fill=(255, 0, 0), width=5)
    draw.line([(x_end, y_end), (x_begin, y_end)], fill=(255, 0, 0), width=5)
    draw.line([(x_begin, y_end), (x_begin, y_begin)], fill=(255, 0, 0), width=5)
    return raw_img, croped_img


def show_crop_full_image(filename, ground_truth_list, crop_list):
    ground_truth_img, croped_ground_truth = draw_rect(filename, ground_truth_list)
    crop_img, croped_crop_img = draw_rect(filename, crop_list)
    plt.subplot(2, 2, 1)
    plt.imshow(ground_truth_img)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(crop_img)
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(croped_ground_truth)
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(croped_crop_img)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # filename = 'FCDB/6931423716_4fb31feac6_c.jpg'
    # ground_truth_list = [3, 130, 797, 195]
    # crop_list = [48, 108, 560, 315]

    # filename = 'FCDB/6356033521_b93c3cf5d6_b.jpg'
    # ground_truth_list = [0, 190, 951, 451]
    # crop_list = [328, 0, 614, 446]

    # filename = 'FCDB/5465125897_db85858f42_b.jpg'
    # ground_truth_list = [0, 0, 661, 967]
    # crop_list = [0, 84, 614, 921]

    # filename = 'FCDB/3528768150_10acc82bb4_o.jpg'
    # ground_truth_list = [216, 124, 615, 474]
    # crop_list = [0, 60, 921, 651]

    # filename = 'FCDB/514599235_3490a79faa_o.jpg'
    # ground_truth_list = [216, 124, 615, 474]
    # crop_list = [38, 30, 742, 423]

    # filename = 'FCDB/8747633714_bea8dd1f01_c.jpg'
    # ground_truth_list = [0, 64, 530, 307]
    # crop_list = [105, 0, 426, 640]

    # filename = 'FCDB/3602624992_491158ee96_b.jpg'
    # ground_truth_list = [460, 229, 266, 236]
    # crop_list = [21, 30, 921, 691]

    # filename = 'FCDB/3318848219_64dd301926_b.jpg'
    # ground_truth_list = [0, 139, 994, 565]
    # crop_list = [84, 0, 921, 691]

    filename = 'FCDB/13091565704_a6bb18d1cb_c.jpg'
    ground_truth_list = [19, 45, 768, 436]
    crop_list = [16, 44, 720, 480]
    show_crop_full_image(filename, ground_truth_list, crop_list)