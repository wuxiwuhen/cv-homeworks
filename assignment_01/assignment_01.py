import cv2
import numpy as np
import os

class picture_augmentationer(object):
    def __init__(self):
        self.name = "This is a picture_augmentationer!"
        
    def img_crop(self, img, x0, y0, x1, y1):
        '''Crop the img with two points'''
        
        height,width = img.shape[0:2]
        assert isinstance(x0,int) and isinstance(y0,int) and isinstance(x1,int) and isinstance(x1,int),"you have non-int arguments!"
        assert x0<height and y0<width-1,"Opps, the shape of img is ({0},{1})you cut them all!".format(height,width)
        return img[x0:x1,y0:y1]
    
    def color_shift(self,img):
        '''Change color randomly'''
        
        channels = cv2.split(img)
        channel_rands = (random.randint(-50, 50) for i in range(3))
        for channel, channel_rand in zip(channels, channel_rands):
            if channel_rand == 0:
                pass
            elif channel_rand > 0:
                lim = 255 - channel_rand
                channel[channel > lim] = 255
                channel[channel <= lim] = (channel_rand + channel[channel <= lim]).astype(img.dtype)
            elif channel_rand < 0:
                lim = 0 - channel_rand
                channel[channel < lim] = 0
                channel[channel >= lim] = (channel_rand + channel[channel >= lim]).astype(img.dtype)
        img_merge = cv2.merge(channels)
        return img_merge
    
    def rotation(self,img, r_center, c_center, angle, scale):
        '''Rotate the img with the center point, angle and scale'''
        
        M = cv2.getRotationMatrix2D((c_center, r_center), angle, scale) # center, angle, scale
        img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return img_rotate
    
    def perspective_transform(self, img, random_margin):
        '''Do perspective transform with img'''
        
        height, width, channels = img.shape

        x1,dx1,x4,dx4,y1,dy1,y2,dy2 = (random.randint(-random_margin, random_margin) for _ in range(8))
        x2,dx2,x3,dx3 = (random.randint(width - random_margin - 1, width - 1) for _ in range(4))
        y3,dy3,y4,dy4 = (random.randint(height - random_margin - 1, height - 1) for _ in range(4))

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, M_warp, (width, height))
        return img_warp
    
    def random_augmentation(self,inPath, outPath, numbers):
        '''Random generate (numbers) pictures with color_shift, rotation, perspective_transform each!'''
        
        assert isinstance(numbers,int) and numbers<=100, "numbers should be an int and less than 100!"
        img = cv2.imread(inPath)
        assert len(img.shape)>2,"your img is not colorful,haha!"
        
        height, width, channels = img.shape
        
        #check outPath,if not exist,make it!
        if not os.path.exists(outPath):
            os.makedirs(outPath)
            
        for i in range(numbers):
            cv2.imwrite(outPath+"/color_shift_"+str(i)+".jpg",self.color_shift(img))
            cv2.imwrite(outPath+"/rotation_"+str(i)+".jpg",self.rotation(img,int(height/2),int(width/2),random.randint(-180,180),0.5+random.random()/2))
            cv2.imwrite(outPath+"/perspective_transform_"+str(i)+".jpg",self.perspective_transform(img,random.randint(0,int(min(height,width)/4))))
        pass

def main():
    inPath = " "
    outPath = " "
    augmentation_case = picture_augmentationer()
    augmentation_case.random_augmentation(inPath, outPath, 100)

if __name__ == '__main__':
    main()