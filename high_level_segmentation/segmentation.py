import numpy as np
import cv2 as cv
import numba
from numba import jit
import queue
from sklearn.cluster import MeanShift, estimate_bandwidth
from PIL import Image


def find_centroid(data, k):
    dim = data.shape[1]
    # 先随机生成一些聚类中心
    cen = np.random.rand(k, dim)

    # 迭代求解
    while True:
        # 只需计算欧式距离的平方
        dis = np.sum((np.expand_dims(data, axis=1) - np.expand_dims(cen, axis=0)) ** 2, axis=-1)

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


def kmeans(im, k=2):
    im = im.astype(np.float64) / 255


    ch = im.shape[-1]
    cen = find_centroid(im.reshape(-1, ch), k)
    nim = im.copy()
    dis = (np.expand_dims(nim, 2) - np.expand_dims(cen, (0, 1))) ** 2
    dis = np.sqrt(np.sum(dis, axis=-1))
    cate = np.argmin(dis, axis=-1)
    nim = cen[cate] * 255
    nim = nim.astype(np.uint8)
    return nim


# 区域增长算法，通过一个栈实现深度优先搜索
def regionGrow(im, threshold=0.03):
    im = im.astype(np.float64) / 255
    dir = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    vis = np.zeros((im.shape[0], im.shape[1]), dtype=np.int32)

    q = queue.LifoQueue()

    cnt = 1
    avcs = []
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if vis[i, j] == 0:
                vis[i][j] = cnt
                q.put(np.array([i, j]))

                avsum = np.zeros_like(im[i, j])
                avcnt = 0
                while not q.empty():
                    u = q.get()
                    avcnt += 1
                    avsum += im[u[0], u[1]]
                    avc = avsum / avcnt
                    np.random.shuffle(dir)
                    an = dir + u
                    an = an[np.where(an[:, 0] >= 0)]
                    an = an[np.where(an[:, 1] >= 0)]
                    an = an[np.where(an[:, 0] < im.shape[0])]
                    an = an[np.where(an[:, 1] < im.shape[1])]
                    for v in an:
                        if np.square(avc - im[v[0], v[1]]).sum() < threshold and vis[v[0], v[1]] == 0:
                            q.put(v)
                            vis[v[0], v[1]] = cnt
                avcs.append(avsum / avcnt)
                cnt += 1

    avcs = np.array(avcs)
    nim = avcs[vis - 1]
    # print(cnt)
    # cnt -= 1
    # for i in range(cnt):
    #     i += 1
    #     nim[np.where(vis == i)] = np.average(nim[np.where(vis == i)], axis=0)

    print(vis)
    return (nim * 255).astype(np.uint8)


# 分水岭算法
def dam(im, EPS=0.05):
    im = im.astype(np.float64) / 255
    print(im.shape)
    gray = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])
    print(gray.shape)

    idx = np.argsort(gray, axis=None, kind='stable')
    idx = np.array([idx // im.shape[1], idx % im.shape[1]]).T
    print(idx.shape)

    vis = np.zeros_like(gray, dtype=np.int32)
    dir = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    last = -1
    cnt = 1

    for u in idx:
        an = dir + u
        an = an[np.where(an[:, 0] >= 0)]
        an = an[np.where(an[:, 1] >= 0)]
        an = an[np.where(an[:, 0] < im.shape[0])]
        an = an[np.where(an[:, 1] < im.shape[1])]
        vset = set()
        for v in an:
            if gray[v[0], v[1]] + EPS < gray[u[0], u[1]] and vis[v[0], v[1]] != 0 and vis[v[0], v[1]] != -1:
                vset.add(vis[v[0], v[1]])
        size = len(vset)
        if size >= 2:  # 如果是分水岭则设为-1
            vis[u[0], u[1]] = -1
        elif size == 1:
            vis[u[0], u[1]] = vset.pop()
        elif size == 0:
            vis[u[0], u[1]] = cnt
            cnt += 1
        if vis[u[0], u[1]] == 0:
            print(u)

    nim = gray.copy()
    nim[np.where(vis == -1)] = 1

    print(vis)
    return (nim * 255).astype(np.uint8)
