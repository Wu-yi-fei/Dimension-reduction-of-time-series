import imageio
import os
import os.path as osp


def img2gif(img_dir, gif_path, duration):
    """
    将多组图像转为gif
    :param img_dir: 包含图片的文件夹
    :param gif_path: 输出的gif的路径
    :param duration: 每张图片切换的时间间隔，与fps的关系：duration = 1 / fps
    :return:
    """
    frames = []
    for idx in sorted(os.listdir(img_dir)):
        img = osp.join(img_dir, idx)
        frames.append(imageio.imread(img))

    imageio.mimsave(gif_path, frames, 'GIF', duration=duration)
    print('Finish changing!')