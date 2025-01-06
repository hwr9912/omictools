import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from skimage.feature.texture import graycomatrix, graycoprops
import cv2
import os
from tqdm import tqdm


class visual():
    def __init__(self, image: np.ndarray, mask: np.ndarray):
        """
        影像组学可视化类
        :param image: CT值影像矩阵，三维矩阵，[slices, width, height]
        :param mask: 01掩膜矩阵
        """
        if not image.ndim == mask.ndim == 3:
            ValueError("Input image and mask must has 3-dimension!")
        if not image.shape == mask.shape:
            ValueError("Shape of image and mask must be the same!")

        self.image_array = image
        self.mask_array = mask
        self.mask_slices = np.array([0 if np.sum(self.mask_array[i, :, :]) == 0 else 1
                                     for i in range(self.image_array.shape[0])])

    def glcm(self, kernel: int = 7):
        # 创建用于存储特征值的数组
        feature_map = np.zeros_like(self.image_array)

        # 遍历图像的每个像素点（考虑边界）
        for i in tqdm(range(self.image_array.shape[0])):
            if np.sum(self.mask_array[i, :, :]) == 0:
                continue
            for j in range(kernel // 2, self.image_array.shape[1] - kernel // 2):
                for k in range(kernel // 2, self.image_array.shape[2] - kernel // 2):
                    if self.mask_array[i, j, k]:  # 仅在掩码内计算特征
                        # 提取局部区域
                        local_patch = self.image_array[
                                      i,
                                      j - kernel // 2:j + kernel // 2 + 1,
                                      k - kernel // 2:k + kernel // 2 + 1
                                      ]

                        # 计算GLCM并提取对比度特征
                        glcm = graycomatrix(local_patch.astype(np.uint8), [1], [0], symmetric=True, normed=True)
                        contrast = graycoprops(glcm, 'contrast')[0, 0]

                        # 将计算的特征值存储在特征映射中
                        feature_map[i, j, k] = contrast

        # 将特征映射应用到感兴趣区域（ROI）
        return feature_map * self.mask_array, self.mask_slices

    def ngtdm(self, kernel: int = 7):
        pad_size = kernel // 2
        ngtdm_kernel = np.ones((kernel, kernel), dtype=np.float32) / (kernel * kernel - 1)
        ngtdm_kernel[pad_size, pad_size] = -1
        ngtdm_matrix = np.zeros_like(self.image_array)

        for i in tqdm(range(self.image_array.shape[0])):
            if np.sum(self.mask_array[i, :, :]) == 0:
                continue
            ngtdm_matrix[i, :, :] = cv2.filter2D(src=self.image_array[i, :, :],
                                                 ddepth=-1, kernel=ngtdm_kernel, borderType=cv2.BORDER_REFLECT)

        return ngtdm_matrix * self.mask_array, self.mask_slices


def triple_weight_map(image: np.ndarray,
                      weight: np.ndarray,
                      image_contrast: tuple = (0, 100),
                      weight_contrast: tuple = (0, 100),
                      transparency: float = 0.25,
                      weight_cmap: str = "hot",
                      show: bool = False,
                      name: str = "triple_weight_visualize.pdf",
                      **kwargs):
    """
    绘制三联矩阵权重热图
    :param image: 原始图像
    :param weight: 权重热图
    :param image_contrast: 原图像阈值（下限，上限）
    :param weight_contrast: 权重图阈值（下限，上限）
    :param transparency: 权重图透明度
    :param weight_cmap: 权重图的色系
    :param name: 图片名称，包含后缀
    :param show: 展示图片
    :param kwargs: plt.savefig的其他参数
    :return: 无输出
    """
    # 使用 gridspec 创建一个包含 4 个子图区域的网格（3 个图像 + 1 个 colorbar）
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    # 第一个子图：显示原始图片 image
    ax1 = plt.subplot(gs[0])
    ax1.imshow(image, cmap='gray', vmin=image_contrast[0], vmax=image_contrast[1])
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 第二个子图：以热图形式显示 weight
    ax2 = plt.subplot(gs[1])
    im = ax2.imshow(weight, cmap=weight_cmap, vmin=weight_contrast[0], vmax=weight_contrast[1])
    ax2.set_title('Weight Heatmap')
    ax2.axis('off')

    # 第三个子图：将 image 作为背景，并在其上叠加 weight 热图（25% 透明度）
    ax3 = plt.subplot(gs[2])
    ax3.imshow(image, cmap='gray', vmin=image_contrast[0], vmax=image_contrast[1])
    ax3.imshow(weight, cmap=weight_cmap, alpha=transparency, vmin=weight_contrast[0], vmax=weight_contrast[1])
    ax3.set_title('Overlay Heatmap on Image')
    ax3.axis('off')

    # 独立的子图：颜色条
    cbar_ax = plt.subplot(gs[3])
    plt.colorbar(im, cax=cbar_ax)

    # 统一调整子图的大小比例
    for ax in [ax1, ax2, ax3]:
        ax.set_aspect('equal', 'box')

    # 显示图形
    plt.tight_layout()

    # 显示图形
    if show:
        plt.show()
    else:
        plt.savefig(name, **kwargs)
    plt.close('all')


if __name__ == "__main__":
    os.chdir(r"D:\Python\machine_learning\SAH_PHH")
    # 加载NIfTI影像和掩码
    image_path = r"D:\Python\machine_learning\SAH_PHH\data\images\CT202310020566.nii.gz"
    mask_path = r"data\masks\CT202310020566.nii.gz"

    image_data = nib.load(image_path)
    mask_data = nib.load(mask_path)

    image_array = np.transpose(image_data.get_fdata(), (2, 0, 1))
    mask_array = np.transpose(mask_data.get_fdata(), (2, 0, 1))

    # 设置滑动窗口大小
    kernel_size = 7

    feature_map_roi, mask_slices = visual(image=image_array, mask=mask_array).ngtdm()
    np.save(r"data/myradiomics.npy", feature_map_roi)
    feature_map_roi = np.load(r"data/myradiomics.npy")
    # 绘制特征热图
    for i in tqdm(range(image_array.shape[0])):
        if mask_slices[i] == 0:
            continue
        plt.figure(figsize=(10, 8))
        sns.heatmap(feature_map_roi[i, :, :], cmap='hot', cbar=True)
        plt.title('Local Radiomics Feature Heatmap')
        plt.savefig(f"figure/CT202310020566.nii.gz_{i}.pdf")
        plt.close('all')
