import cv2
import numpy as np

def medianBlur1(img, kernel, padding_way):
    res = np.zeros_like(img)
    height, width = img.shape
    half_k = int((kernel-1)/2)
    img_add_padding = np.zeros((height + 2 * half_k, width + 2 * half_k), dtype=int)
    if padding_way == "REPLICA":
        #四条边区域
        img_add_padding[half_k:-half_k, 0:half_k] = img[:, 0].reshape(height, 1)
        img_add_padding[half_k:-half_k, -half_k:] = img[:, -1].reshape(height, 1)
        img_add_padding[0:half_k, half_k:-half_k] = img[0, :].reshape(1, width)
        img_add_padding[-half_k:, half_k:-half_k] = img[-1, :].reshape(1, width)
        
        #四个角区域
        img_add_padding[0:half_k, 0:half_k] = img[0, 0]
        img_add_padding[0:half_k, -half_k:] = img[0,-1]
        img_add_padding[-half_k:, 0:half_k] = img[-1,0]
        img_add_padding[-half_k:, -half_k:] = img[-1,-1]
        
        #主体
        img_add_padding[half_k:-half_k, half_k:-half_k] = img
    elif padding_way == "ZERO":
        img_add_padding[half_k:-half_k, half_k:-half_k] = img
    else:
        return "Not support for this kind of " + padding_way
    
    for i in range(half_k, height + half_k):
        for j in range(half_k, width + half_k):
            res[i-half_k,j-half_k] = np.median(img_add_padding[i - half_k:i + half_k + 1,j - half_k:j + half_k + 1])
    return res

def find_Median_Of_Hist(hist,thread):
    res = 0
    for i in range(256):
        res += hist[i]
        if res>thread:
            return i
    return 255
    
def medianBlur2(img, kernel, padding_way):
    res = np.zeros_like(img)
    height, width = img.shape
    half_k = int((kernel-1)/2)
    img_add_padding = np.zeros((height + 2 * half_k, width + 2 * half_k), dtype=int)
    if padding_way == "REPLICA":
        #四条边区域
        img_add_padding[half_k:-half_k, 0:half_k] = img[:, 0].reshape(height, 1)
        img_add_padding[half_k:-half_k, -half_k:] = img[:, -1].reshape(height, 1)
        img_add_padding[0:half_k, half_k:-half_k] = img[0, :].reshape(1, width)
        img_add_padding[-half_k:, half_k:-half_k] = img[-1, :].reshape(1, width)
        
        #四个角区域
        img_add_padding[0:half_k, 0:half_k] = img[0, 0]
        img_add_padding[0:half_k, -half_k:] = img[0,-1]
        img_add_padding[-half_k:, 0:half_k] = img[-1,0]
        img_add_padding[-half_k:, -half_k:] = img[-1,-1]
        
        #主体
        img_add_padding[half_k:-half_k, half_k:-half_k] = img
    elif padding_way == "ZERO":
        img_add_padding[half_k:-half_k, half_k:-half_k] = img
    else:
        return "Not support for this kind of " + padding_way
    
    #采用直方图统计方法----测试了下，比np.median方法还慢很多啊，是我的打开方式不对吗？！
    thread = int(kernel * kernel/2)
    
    for i in range(half_k, height + half_k):
        hist = np.zeros(256)
        for k in img_add_padding[i - half_k:i + half_k + 1,0:2 * half_k + 1].ravel():
            hist[k] += 1
        res[i-half_k,0] = find_Median_Of_Hist(hist, thread)
        
        for j in range(half_k+1, width + half_k):
            for k in img_add_padding[i - half_k:i + half_k + 1, j - half_k - 1]:
                hist[k] -= 1
            for k in img_add_padding[i - half_k:i + half_k + 1, j + half_k]:
                hist[k] += 1
            res[i-half_k,j-half_k] = find_Median_Of_Hist(hist, thread)
    return res

def main():
    inPath = "./assignment_02/data/lena.jpg"
    img = cv2.imread(inPath,0)
    img_median_01 = medianBlur1(img,5,"REPLICA")
    img_median_02 = medianBlur2(img,5,"REPLICA")
    img_median_03 = cv2.medianBlur(img,5)

    #test
    assert (img_median_01==img_median_02).all()
    assert (img_median_01==img_median_03).all()

    print("code run as expected!")

if __name__ == '__main__':
    main()
