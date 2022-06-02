import numpy as np

class Lattice:
    def __init__(self, indices, l, m, n):
        self.space = 0.01
        self.l,self.m,self.n= l,m,n
        self.init_knots()

        min_x,min_y,min_z = np.min(indices,axis=0)[0:3]-self.space
        max_x,max_y,max_z = np.max(indices,axis=0)[0:3]+self.space

        self.lattice_origin = [min_x,min_y,min_z]
        self.lattice_size = [max_x-min_x,max_y-min_y,max_z-min_z]

        self.control_points = np.mgrid[min_x:max_x:self.l*1j,min_y:max_y:self.m*1j,min_z:max_z:self.n*1j].reshape(3,-1).T

        ind=np.argsort(self.control_points[:,1])

        self.vertices = []
        for cp in self.control_points:
            #print(cp)
            lv = LatticeVertex(cp,len(self.vertices))
            self.vertices.append(lv)
        self.vertices_copy = np.array(self.vertices)[ind]
        self.planes = np.split(np.array(self.vertices_copy), self.m)
        ind_v=np.argsort(self.control_points[:,0])
        self.vertices_copy = np.array(self.vertices)[ind_v]
        self.v_planes = np.split(np.array(self.vertices_copy), self.l)

    def init_knots(self):
        self.l_knots = np.arange(self.l-2)
        self.l_knots=np.insert(self.l_knots,0,[0,0])
        self.l_knots=np.append(self.l_knots,[self.l-3,self.l-3])
        self.m_knots = np.arange(self.m-2)
        self.m_knots=np.insert(self.m_knots,0,[0,0])
        self.m_knots=np.append(self.m_knots,[self.m-3,self.m-3])
        self.n_knots = np.arange(self.n-2)
        self.n_knots=np.insert(self.n_knots,0,[0,0])
        self.n_knots=np.append(self.n_knots,[self.n-3,self.n-3])
        self.knot_size = [self.l_knots[-1],self.m_knots[-1],self.n_knots[-1]]

class LatticeVertex:
    def __init__(self,pos,index):
        self.position = pos
        self.index = index
        self.selected = False
        self.screen_coord = None

    def get_color(self):
        color = [0.97,0.71,0]
        if self.selected:
            color = [0.8,0.0,0.0]
        return color

    def toggle_selection(self):
        if self.selected:
            self.selected = False
        else:
            self.selected = True

    def set_screne_coord(self, screen_coord):
        self.screen_coord = screen_coord

    def in_me(self,coord):
        x_dif = abs(coord[0]-self.screen_coord[0])
        y_dif = abs(coord[1]-self.screen_coord[1])
        if x_dif < 10 and y_dif < 10:
            return True
        else:
            return False
