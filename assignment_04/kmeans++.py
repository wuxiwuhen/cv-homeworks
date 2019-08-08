import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def assignment(df, centroids, colmap):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

def update(df, centroids):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

########--------------此部分函数用于k-means++初始化
def get_dist_from_nearest_centroids(df, centroids):
    res = []
    for i in range(len(df)):
        distances =[np.sqrt((df['x'][i] - value[0])**2 + (df['y'][i] - value[1])**2) for value in centroids.values()]
        res.append(min(distances))
    return res
    
def get_probility_for_top_n(probility_for_each_point):
    for i in range(1,len(probility_for_each_point)):
        probility_for_each_point[i] = probility_for_each_point[i] + probility_for_each_point[i-1]
    return probility_for_each_point

def get_loc_for_ci(probility_for_top_n, rand_probility):
    for i in range(len(probility_for_top_n)):
        if rand_probility <= probility_for_top_n[i]:
            return i

def get_k_Centroids_from_data(df, k):
    c1 = np.random.randint(len(df))
    centroids = {0:[df['x'][c1], df['y'][c1]]}
    k -= 1
    i = 0
    while k > 0:
        dist_from_nearest_centroids = np.array(get_dist_from_nearest_centroids(df, centroids))
        probility_for_each_point =  dist_from_nearest_centroids/sum(dist_from_nearest_centroids)
        probility_for_top_n = get_probility_for_top_n(probility_for_each_point)
        rand_probility = np.random.random()
        ci = get_loc_for_ci(probility_for_top_n, rand_probility)
        i += 1
        centroids[i] = [df['x'][ci],df['y'][ci]]
        k -= 1
    return centroids
########--------------此部分函数用于k-means++初始化
    
def main():
    # step 0.0: generate source data
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })
    #np.random.seed(200)    # in order to fix the random centorids
    k = 3
    # centroids[i] = [x, y]

    # 采用k-means++的思想初始化k个中心点
    centroids = get_k_Centroids_from_data(df, k)

    # step 0.2: assign centroid for each source data
    # colmap = {0: 'r', 1: 'g', 2: 'b', 3: 'm', 4: 'c'}
    colmap = {0: 'r', 1: 'g', 2: 'b'}
    df = assignment(df, centroids, colmap)

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()

    for i in range(10):
        key = cv2.waitKey()
        plt.close()

        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(df, centroids)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

        df = assignment(df, centroids, colmap)

        if closest_centroids.equals(df['closest']):
            break


if __name__ == '__main__':
    main()
