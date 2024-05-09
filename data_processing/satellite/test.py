import urllib.request

import os
import cv2
import numpy as np
import random
import re
import math
import find_maxpic
from tqdm import tqdm
import latlon_transform as lt

def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)
    if direction == 'E' or direction == 'N':
        dd *= -1
    return dd


def parse_dms(dms):
    parts = re.split('[^\d\w]+', dms)
    lat = dms2dd(parts[0], parts[1], parts[2], parts[3])
    return (lat)


count = 0  # 用于getimg异常次数过多退出计数

agents = [
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/532.5 (KHTML, like Gecko) Chrome/4.0.249.0 Safari/532.5',
    'Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US) AppleWebKit/532.9 (KHTML, like Gecko) Chrome/5.0.310.0 Safari/532.9',
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/534.7 (KHTML, like Gecko) Chrome/7.0.514.0 Safari/534.7',
    'Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/534.14 (KHTML, like Gecko) Chrome/9.0.601.0 Safari/534.14',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.14 (KHTML, like Gecko) Chrome/10.0.601.0 Safari/534.14',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.20 (KHTML, like Gecko) Chrome/11.0.672.2 Safari/534.20',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.27 (KHTML, like Gecko) Chrome/12.0.712.0 Safari/534.27',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/13.0.782.24 Safari/535.1']


# 经纬度反算切片行列号 3857坐标系
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)


# 下载图片
def getimg(Tpath, Spath, x, y):
    global count
    count = 0  # 清零
    try:
        f = open(Spath, 'wb')
        req = urllib.request.Request(Tpath)
        req.add_header('User-Agent', random.choice(agents))  # 换用随机的请求头
        pic = urllib.request.urlopen(req, timeout=60)
        f.write(pic.read())
        f.close()
        # print(str(x) + '_' + str(y) + '下载成功')
    except Exception:
        print(str(x) + '_' + str(y) + '下载失败,重试')  # 存在一直失败不跳出的bug
        getimg(Tpath, Spath, x, y)
        count = count + 1
        if count > 100:
            return


def download(zoom, LTlat, LTlon, RBlat, RBlon, rootDir,id):
    lefttop = deg2num(LTlat, LTlon, zoom)  # 下载切片的左上角角点
    rightbottom = deg2num(RBlat, RBlon, zoom)

    print("共{:d}张图像".format((lefttop[0] - rightbottom[0]) * (lefttop[1] - rightbottom[1])))

    for x in range(lefttop[0], rightbottom[0]):

        path = rootDir + "\\process\\" + str(id) + "\\" + str(x)  # 文件夹检查

        if not os.path.exists(path):
            os.makedirs(path)
        with tqdm(range(lefttop[1], rightbottom[1])) as pbar:
            for y in pbar:
                pbar.set_description(f"Schedule {x - lefttop[0] + 1}/{rightbottom[0] - lefttop[0]}")
                tilepath = "http://mt0.google.com/vt?lyrs=s&x=" + str(x) + "&y=" + str(y) + "&z=" + str(zoom)
                filepath = path + "\\" + str(y) + ".png"  # 文件检查
                if not os.path.isfile(filepath):
                    getimg(tilepath, os.path.join(path, str(y) + ".png"), x, y)

    print('地图下载完成')


def merge(x1, y1, x2, y2, z, path, id):
    row_list = list()
    for i in range(x1, x2 + 2):
        col_list = list()
        for j in range(y1, y2 + 2):
            path_img = path + "\\process\\" + str(id) + "\\{i}\\{j}.png".format(i=i, j=j, z=z)
            img = cv2.imread(path_img)
            col_list.append(img)
        k = np.vstack(col_list)
        row_list.append(k)
    result = np.hstack(row_list)
    cv2.imwrite(path + "\\process\\"+ str(id) +"\\"+str(id)+".png", result)
    cv2.imwrite(path + "\\result\\"+ str(id) + ".png", result)
    print("地图合并完成，保存为：{:s}".format(path + "\\result\\" + str(id) + ".png", result))


if __name__ == "__main__":
    rootDir = "D:\\satellite"
    resultDir = "D:\\satellite\\result"
    # 瓦片地图的放大倍数
    zoom = 20
    #用于定义图片/区域编号

    #判断从哪个位置开始处理
    turn = find_maxpic.find_max_number_file('D:\\satellite\\result') + 1

    for id in range(turn,677):
        # 这里定义下载范围
        std_lat = 30.728541
        std_lon = 103.990737
        LT_lat = std_lat - ( (id - 1) // 26 ) *0.005
        LT_lon = std_lon + ((id - 1) % 26) * 0.005
        RB_lat = LT_lat-0.005
        RB_lon = LT_lon+0.005
        # 打印坐标
        print("当前在处理：")
        print(id,LT_lat, LT_lon, RB_lat, RB_lon)
        # delta_lat = LT_lat - RB_lat
        # delta_lon = RB_lon - LT_lon

        # if zoom > 15:
        #     LT_lat = LT_lat - delta_lat * 1.0 / 4
        #     LT_lon = LT_lon + delta_lon * 1.0 / 4
        #     RB_lat = RB_lat + delta_lat * 1.0 / 4
        #     RB_lon = RB_lon - delta_lon * 1.0 / 4

        download(zoom, LT_lat, LT_lon, RB_lat, RB_lon, rootDir,id)

        print("开始合并地图......")
        # 合并保存为大图
        lefttop = deg2num(LT_lat, LT_lon, zoom)  # 下载切片的左上角角点
        rightbottom = deg2num(RB_lat, RB_lon, zoom)
        merge(lefttop[0], lefttop[1], rightbottom[0]-2, rightbottom[1]-2, zoom, rootDir,id)
