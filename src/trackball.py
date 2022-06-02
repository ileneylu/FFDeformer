import numpy as np
import math

class Trackball:
    def __init__(self, width, height):
        self.radius = min(width,height)/2
        self.center = (width/2,height/2)

    def get_hemisphere_coord(self,x,y):
        h_x = x-self.center[0]
        h_y = y-self.center[1]
        h_x_sq = h_x**2
        h_y_sq = h_y**2
        xy_sum = h_x_sq + h_x_sq
        radius_sq = self.radius**2
        if xy_sum > radius_sq:
            ratio = xy_sum/radius_sq
            h_x = math.sqrt(h_x_sq/ratio)
            h_y = math.sqrt(h_y_sq/ratio)
            xy_sum = radius_sq
        h_z = math.sqrt(radius_sq - xy_sum)
        return (h_x,-h_y,h_z)

    def get_axis_angle(self,p1,p2):
        p1_arr = self.normalize(np.array(p1))
        p2_arr = self.normalize(np.array(p2))
        n = np.cross(p1_arr,p2_arr)
        norm = np.linalg.norm(n)
        if norm == 0:
            return [1,0,0], 0
        n = self.normalize(n)
        diff_sum = (p2_arr[0]-p1_arr[0])**2+(p2_arr[1]-p1_arr[1])**2 + (p2_arr[2]-p1_arr[2])**2
        angle = np.radians(40*math.sqrt(diff_sum))
        return n,angle

    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm == 0:
           return v
        return v / norm
