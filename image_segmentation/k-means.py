import numpy as np
import cv2 as cv


def kmeans(data, k):
    dim = data.shape[1]
    # 先随机生成一些聚类中心
    cen = np.random.rand(k, dim)

    # 迭代求解
    while True:
        # 计算欧式距离
        dis = (np.expand_dims(data, axis=1) - np.expand_dims(cen, axis=0)) ** 2
        dis = np.sqrt(np.sum(dis, axis=-1))

        # 计算类别
        cate = np.argmin(dis, axis=1)
        ncen = cen.copy()

        for i in range(k):
            # 计算同类别的新均值
            ncen[i] = np.average(data[np.where(cate == i)], axis=0)
            # 如果有任何点都不属于的情况，则新随机生成一个点
            if np.any(np.isnan(ncen[i])):
                ncen[i] = np.random.rand(dim)
        # 如果两次结果一样，则达到停止循环条件
        if np.all(np.abs(cen - ncen) < 1e-4):
            break
        else:
            cen = ncen
    # 返回聚类中心
    return cen


def segmentation(path, k=2):
    im = cv.imread(path).astype(np.float64) / 255



    cen = kmeans(im.reshape(-1, 3), k)
    nim = im.copy()
    dis = (np.expand_dims(nim, 2) - np.expand_dims(cen, (0, 1))) ** 2
    dis = np.sqrt(np.sum(dis, axis=-1))
    cate = np.argmin(dis, axis=-1)
    nim = cen[cate]
    cv.imshow('nim', nim)
    cv.imwrite('out{}.jpg'.format(k), nim * 255)
    cv.waitKey(0)


for x in range(2, 11):
    segmentation('in.jpg', x)
