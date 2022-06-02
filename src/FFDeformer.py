import numpy as np
from scipy import spatial
import math
import kivy
kivy.require('1.11.0')

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.transformation import Matrix
from kivy.graphics.opengl import *
from kivy.graphics import *
from kivy.resources import resource_find
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.dropdown import DropDown
from kivy.config import Config
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput

from objloader import ObjFile
from functools import partial
from latticebuilder import *
from trackball import *

Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '900')
Config.write()

class Renderer(Widget):
    def __init__(self, **kwargs):
        super(Renderer, self).__init__(**kwargs)
        self.vertex_mesh = list(ObjFile(resource_find("primitives/cp.obj")).objects.values())[0].vertices
        self.vertex_mesh = np.array(self.vertex_mesh).reshape(-1,9)
        self.vertex_mesh_fmt = [
            (b'v_pos', 3, 'float'),
            (b'v_color', 3, 'float'),
            (b'v_normal', 3, 'float')]
        self.canvas = RenderContext(compute_normal_mat=True)
        self.canvas.shader.source = resource_find('simple.glsl')
        self.init_graphics()
        self.scene = None
        self.moving = False
        self.mode = 'display'
        self.lattice_selected = 0
        self.l, self.m, self.n = 4,4,4
        self.lattice_visible = True
        self.object_visible = True
        self.undo_stack = []
        self.redo_stack = []

    def redo(self):
        if len(self.redo_stack) > 0:
            self.undo_stack.append(np.copy(self.lattice.control_points))
            self.lattice.control_points = self.redo_stack.pop()
            for lcp in self.lattice.vertices:
                lcp.position = self.lattice.control_points[lcp.index]
                self.update_lattice_mesh(lcp.index)
            self.update_object_vertices()

    def undo(self):
        if len(self.undo_stack) > 0:
            self.redo_stack.append(np.copy(self.lattice.control_points))
            self.lattice.control_points = self.undo_stack.pop()
            for lcp in self.lattice.vertices:
                lcp.position = self.lattice.control_points[lcp.index]
                self.update_lattice_mesh(lcp.index)
            self.update_object_vertices()

    def init_graphics(self):
        self.cam_pos = [0.0,0.0,-4.0]
        self.mv_mat = Matrix().look_at(self.cam_pos[0],self.cam_pos[1],self.cam_pos[2], 0.0, 0.0, 0.0, 0.0, 1.0,0.0)
        self.rotation = Matrix().rotate(np.radians(0),0,1,0)
        self.window_width, self.window_height = Window.size
        self.fov = 1

    def updata_obj_dir(self,dir):
        self.mv_mat = Matrix().look_at(self.cam_pos[0],self.cam_pos[1],self.cam_pos[2], 0.0, 0.0, 0.0, 0.0, 1.0,0.0)
        if dir == 'back':
            rotation = Matrix().rotate(np.pi,0.0,1.0,0.0)
            self.mv_mat.set(self.mv_mat.multiply(rotation))
        elif dir == 'left':
            rotation = Matrix().rotate(np.pi/2,0.0,1.0,0.0)
            self.mv_mat.set(self.mv_mat.multiply(rotation))
        elif dir == 'right':
            rotation = Matrix().rotate(-np.pi/2,0.0,1.0,0.0)
            self.mv_mat.set(self.mv_mat.multiply(rotation))
        elif dir == 'top':
            rotation = Matrix().rotate(-np.pi/2,1.0,0.0,0.0)
            self.mv_mat.set(self.mv_mat.multiply(rotation))
        elif dir == 'bottom':
            rotation = Matrix().rotate(np.pi/2,1.0,0.0,0.0)
            self.mv_mat.set(self.mv_mat.multiply(rotation))
        self.update_glsl(True)


    def init_lattice(self,vertices):
        vertices = np.array(vertices).reshape(-1,9)[:,:3]
        self.lattice = Lattice(vertices, self.l, self.m, self.n)

    def render_scene(self):
        with self.canvas:
            self.cb = Callback(self.setup_gl_context)
            if self.object_visible:
                self.setup_scene()
            if self.lattice_visible:
                self.setup_lattice()
            self.cb = Callback(self.reset_gl_context)
            self.update_glsl(True)

    def setup_scene(self):
        Color(1, 1, 1, 1)
        if self.scene != None:
            m = list(self.scene.objects.values())[0]
            UpdateNormalMatrix()
            self.mesh = Mesh(
                vertices=m.vertices,
                indices=m.indices,
                fmt=m.vertex_format,
                mode='triangles',
            )

    def setup_lattice(self):
        self.lattice_mesh = []
        UpdateNormalMatrix()
        for i,cp in enumerate(self.lattice.vertices):
            lvertices = []
            lindices = []
            color = cp.get_color()
            for j,cp_mesh in enumerate(self.vertex_mesh):
                value =cp.position+cp_mesh[:3]*0.02
                lvertices.append(value)
                lvertices.append(color)
                lvertices.append(cp_mesh[6:9])
                lindices.extend([3*j,3*j+1,3*j+2])
            vertices=np.array(lvertices).flatten().astype(np.float32)
            mesh = Mesh(
                vertices=vertices,
                indices=lindices,
                fmt=self.vertex_mesh_fmt,
                mode='triangles',
            )
            self.lattice_mesh.append(mesh)

    def load_obj(self,obj):
        self.canvas.clear()
        self.init_graphics()
        self.cur_obj = obj
        if obj == 'cube':
            self.scene = ObjFile(resource_find("primitives/cube_min.obj"))
        elif obj == 'sphere':
            self.scene = ObjFile(resource_find("primitives/sphere.obj"))
        elif obj == 'cylinder':
            self.scene = ObjFile(resource_find("primitives/cylinder.obj"))
        elif obj == 'torus':
            self.scene = ObjFile(resource_find("primitives/doughnut.obj"))
        else:
            self.scene = ObjFile(resource_find(obj))
        m = list(self.scene.objects.values())[0]
        self.init_lattice(m.vertices)
        self.scene.init_vertex_info(self.lattice.lattice_origin, self.lattice.lattice_size, self.lattice.knot_size)
        self.init_vertex_basis_value()
        self.render_scene()
        self.init_trackball()

    def perpendicular_vector(self, v):
        epsilon = 0.1
        if abs(v[1]) < epsilon and abs(v[2]) < epsilon:
                return np.cross(v, [0, 1, 0])
        return np.cross(v, [1, 0, 0])

    def init_vertex_basis_value(self):
        l_knots = self.lattice.l_knots
        m_knots = self.lattice.m_knots
        n_knots = self.lattice.n_knots
        num_control_points = len(self.lattice.vertices)
        m_size = self.lattice.m
        n_size = self.lattice.n
        ori = self.lattice.lattice_origin
        lattice_size = self.lattice.lattice_size
        knot_size = self.lattice.knot_size

        index = 0
        for v_index,vertex in enumerate(self.scene.vertices):
            i,j,k = vertex.segment
            s,t,u = vertex.local_coord
            n_index = self.scene.dict_vertex_normal.get(str(v_index+1))
            n = self.scene.normals[int(n_index)-1]
            vec_1 = self.perpendicular_vector(n)
            vec_1 = vec_1/np.linalg.norm(vec_1)
            vec_2 = np.cross(n,vec_1)
            vec_2 = vec_2/np.linalg.norm(vec_2)
            ref1 = vertex.position + 0.01 * vec_1
            ref2 = vertex.position + 0.01 * vec_2
            ref1_local = ref1 - ori
            ref1_local = ref1_local/lattice_size*knot_size
            ref2_local = ref2 - ori
            ref2_local = ref2_local/lattice_size*knot_size

            vertex.b = np.zeros(num_control_points)
            vertex.b_vec1 = np.zeros(num_control_points)
            vertex.b_vec2 = np.zeros(num_control_points)
            knot_index_l = self.get_knot_index(s)
            knot_index_m = self.get_knot_index(t)
            knot_index_n = self.get_knot_index(u)

            basis_l = self.get_basis_function(s,knot_index_l,l_knots)
            basis_m = self.get_basis_function(t,knot_index_m,m_knots)
            basis_n = self.get_basis_function(u,knot_index_n,n_knots)

            basis_vec1_l = self.get_basis_function(ref1_local[0],knot_index_l,l_knots)
            basis_vec1_m = self.get_basis_function(ref1_local[1],knot_index_m,m_knots)
            basis_vec1_n = self.get_basis_function(ref1_local[2],knot_index_n,n_knots)

            basis_vec2_l = self.get_basis_function(ref2_local[0],knot_index_l,l_knots)
            basis_vec2_m = self.get_basis_function(ref2_local[1],knot_index_m,m_knots)
            basis_vec2_n = self.get_basis_function(ref2_local[2],knot_index_n,n_knots)

            for l in range(4):
                for m in range(4):
                    for n in range(4):
                        cp_index = int((i+l)*m_size*n_size + (j+m)*n_size + (k+n))
                        vertex.b[cp_index] = basis_l[l]*basis_m[m]*basis_n[n]
                        vertex.b_vec1[cp_index] = basis_vec1_l[l]*basis_vec1_m[m]*basis_vec1_n[n]
                        vertex.b_vec2[cp_index] = basis_vec2_l[l]*basis_vec2_m[m]*basis_vec2_n[n]


            vertex.b_inv = vertex.b / np.linalg.norm(vertex.b)
            vertex.b_inv = np.array([vertex.b_inv]).T

    def init_trackball(self):
        self.trackball = Trackball(self.window_width,self.window_height)

    def update_glsl(self,light):
        asp = float(self.window_width) / float(self.window_height)

        self.proj = Matrix().view_clip(-asp*self.fov, asp*self.fov, -1*self.fov, 1*self.fov, 1, 600, 1)
        self.canvas['modelview_mat'] = self.mv_mat
        self.canvas['projection_mat'] = self.proj
        self.set_lattice_screen_coord()
        self.set_obj_screen_coord()
        if light:
            self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
            self.canvas['ambient_light'] = (0.1, 0.1, 0.1)

    def set_lattice_screen_coord(self):
        for lcp in self.lattice.vertices:
            screen_coord = Matrix().project(lcp.position[0],lcp.position[1],lcp.position[2],self.mv_mat,self.proj,0,0, self.window_width,self.window_height)
            lcp.set_screne_coord(screen_coord)

    def set_obj_screen_coord(self):
        self.scene.screen_coord = []
        for vertex in self.scene.vertices:
            screen_coord = Matrix().project(vertex.position[0],vertex.position[1],vertex.position[2],self.mv_mat,self.proj,0,0, self.window_width,self.window_height)
            self.scene.screen_coord.append(screen_coord)
        self.scene.screen_coord = np.array(self.scene.screen_coord)
        self.scene.kdtree = spatial.KDTree(self.scene.screen_coord[:,:2])

    def update_lattice_mesh(self, index):
        cp = self.lattice.vertices[index]
        lvertices = []
        for j,cp_mesh in enumerate(self.vertex_mesh):
            value =cp.position+cp_mesh[:3]*0.02
            lvertices.append(value)
            lvertices.append(cp.get_color())
            lvertices.append(cp_mesh[6:9])
        vertices=np.array(lvertices).flatten().astype(np.float32)
        self.lattice_mesh[index].vertices = vertices
        self.update_glsl(True)

    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def transform_point(self, mat, x, y, z, t):
        tx = x * mat[0] + y * mat[4] + z * mat[ 8] + mat[12];
        ty = x * mat[1] + y * mat[5] + z * mat[ 9] + mat[13];
        tz = x * mat[2] + y * mat[6] + z * mat[10] + mat[14];
        tt = x * mat[3] + y * mat[7] + z * mat[11] + mat[15];
        return [tx, ty, tz, tt]

    def project(self,objx,objy,objz,model,proj, vx, vy, vw, vh):
        point = self.transform_point(model,objx, objy, objz, 1.0)
        point = self.transform_point(proj,point[0],point[1],point[2],point[3])
        if point[3] == 0:
            return None

        point[0] /= point[3]
        point[1] /= point[3]
        point[2] /= point[3]
        winx = vx + (1 + point[0]) * vw / 2.
        winy = vy + (1 + point[1]) * vh / 2.
        winz = (1 + point[2]) / 2.

        return (winx, winy, winz)

    def get_knot_index(self, u):
        return math.floor(u) + 4

    def get_basis_function(self, u, ki, knots):
        basis_degree = 4
        N = np.zeros((basis_degree,basis_degree+2))
        D = np.zeros(basis_degree+2)
        N[0,basis_degree] = 1
        diff = ki-3
        knot_vector = np.insert(knots, 0,[0,0])
        knot_vector = np.append(knot_vector, [knot_vector[-1],knot_vector[-1]])
        for i in range(1, basis_degree):
            for j in range(basis_degree+1):
                if N[i-1,j] != 0 and (u-knot_vector[j-1+diff])!= 0 and (knot_vector[i+j-1+diff]-knot_vector[j-1+diff]) != 0:
                    N[i,j] = N[i,j] + N[i-1,j]*(u-knot_vector[j-1+diff])/(knot_vector[i+j-1+diff]-knot_vector[j-1+diff])
                if N[i-1,j+1] != 0 and (knot_vector[i+j+diff]-u)!= 0 and (knot_vector[i+j+diff]-knot_vector[j+diff]) != 0:
                    N[i,j] = N[i,j] + N[i-1,j+1]*(knot_vector[i+j+diff]-u)/(knot_vector[i+j+diff]-knot_vector[j+diff])

        return N[i,1:-1]

    def on_touch_down(self,touch):
        if self.scene != None:
            if self.collide_point(touch.pos[0], touch.pos[1]):
                if self.mode == 'select':
                    if touch.is_double_tap:
                        isLattice = False
                        selected_cp = []
                        selected_cp_z = []
                        for lcp in self.lattice.vertices:
                            in_cur_cp = lcp.in_me(touch.pos)
                            if in_cur_cp:
                                selected_cp.append(lcp)
                                selected_cp_z.append(lcp.screen_coord[2])
                            isLattice = isLattice or in_cur_cp
                        if len(selected_cp_z) != 0:
                            j = selected_cp_z.index(min(selected_cp_z))
                            selected_cp[j].toggle_selection()
                            if selected_cp[j].selected:
                                self.lattice_selected += 1
                            else:
                                self.lattice_selected -= 1
                            self.update_lattice_mesh(selected_cp[j].index)
                    else:
                        self.start = touch.pos
                self.moving = True
                if self.mode == 'display':
                    self.p1 = self.trackball.get_hemisphere_coord(touch.pos[0], touch.pos[1])
                if self.mode == 'edit':
                    self.start = touch.pos
                    self.mv_arr = self.mv_mat.get()
                    self.proj_arr = self.proj.get()
                    self.mv_i_arr = self.mv_mat.inverse().get()
                    self.proj_i_arr = self.proj.inverse().get()
                    self.moved = False
                    self.undo_stack.append(np.copy(self.lattice.control_points))
                if self.mode == 'direct':
                    self.start = touch.pos
                    self.mv_arr = self.mv_mat.get()
                    self.proj_arr = self.proj.get()
                    self.mv_i_arr = self.mv_mat.inverse().get()
                    self.proj_i_arr = self.proj.inverse().get()
                    d,indices = self.scene.kdtree.query(np.array(touch.pos),k =5,distance_upper_bound=50)
                    indices = indices[indices != self.scene.num_vertex]
                    if indices.size == 0:
                        self.selected_mesh_vertex = None
                    else:
                        i = indices[self.scene.screen_coord[indices,2].argmin()]
                        print('vertex index: ' + str(i))
                        self.selected_mesh_vertex = self.scene.vertices[i]
                    self.moved = False
                if self.mode == 'bend' or self.mode == 'twist':
                    self.start = touch.pos
                    self.mv_arr = self.mv_mat.get()
                    self.proj_arr = self.proj.get()
                    self.mv_i_arr = self.mv_mat.inverse().get()
                    self.proj_i_arr = self.proj.inverse().get()
                    self.start = touch.pos
                    self.moving_path = []
                    self.plane_y = np.zeros(self.lattice.m)
                    self.plane_x = np.zeros(self.lattice.l)
                    self.plane_z = 0
                    for index, plane in enumerate(self.lattice.planes):
                        pos=[o.position for o in plane]
                        pos = np.array(pos)
                        #print(pos)
                        i=np.argmin(pos[:,-1])
                        self.plane_y[index] = plane[i].screen_coord[1]
                        self.plane_z = plane[i].screen_coord[2]
                    self.nearest_plane_index = np.abs(self.plane_y - touch.pos[1]).argmin()
                    for index, plane in enumerate(self.lattice.v_planes):
                        pos=[o.position for o in plane]
                        pos = np.array(pos)
                        #print(pos)
                        i=np.argmin(pos[:,-1])
                        self.plane_x[index] = plane[i].screen_coord[0]
                        self.plane_z = plane[i].screen_coord[2]
                    self.nearest_v_plane_index = np.abs(self.plane_x - touch.pos[0]).argmin()
                    self.moving_path.append(self.start)
                    #print(self.nearest_plane_index)


    def on_touch_move(self,touch):
        if self.collide_point(touch.pos[0], touch.pos[1]):
            if self.moving:
                if self.mode == 'display':
                    self.p2 = self.trackball.get_hemisphere_coord(touch.pos[0], touch.pos[1])
                    axis,angle = self.trackball.get_axis_angle(self.p1,self.p2)
                    rotation = Matrix().rotate(angle,axis[0],axis[1],axis[2])

                    self.mv_mat.set(self.mv_mat.multiply(rotation))
                    self.p1 = self.p2
                    self.update_glsl(True)
                if self.mode == 'edit':
                    self.cur = touch.pos
                    diff = [self.cur[0]-self.start[0],self.cur[1]-self.start[1]]
                    selected_vertices = []
                    selected_vertices_z = []
                    for lcp in self.lattice.vertices:
                        if lcp.selected:
                            selected_vertices.append(lcp)
                            selected_vertices_z.append(lcp.position[2])
                    closest_z = min(selected_vertices_z)
                    if len(selected_vertices) != 0:
                        self.redo_stack.clear()
                    else:
                        self.undo_stack.pop()
                    for lcp in selected_vertices:
                        point = self.transform_point(self.mv_arr,lcp.position[0],lcp.position[1],closest_z,1.0)
                        point = self.transform_point(self.proj_arr,point[0],point[1],point[2],point[3])
                        x_clip = diff[0]*2/self.window_width
                        y_clip = diff[1]*2/self.window_height
                        x_clip = x_clip * point[3] + point[0]
                        y_clip = y_clip * point[3] + point[1]
                        point = self.transform_point(self.proj_i_arr,x_clip,y_clip,point[2],point[3])
                        point = self.transform_point(self.mv_i_arr,point[0],point[1],point[2],point[3])
                        lcp.position = np.array([point[0],point[1],lcp.position[2]])
                        self.lattice.control_points[lcp.index] = lcp.position
                        self.update_lattice_mesh(lcp.index)
                    self.start = self.cur
                    self.moved = True
                if self.mode == 'direct':
                    self.moved = True
                if self.mode == 'bend' or self.mode == 'twist':
                    self.moving_path.append(touch.pos)

    def update_object_vertices(self):
        m_size = self.lattice.m
        n_size = self.lattice.n
        for index,vertex in enumerate(self.scene.vertices):
            pos = np.matmul(vertex.b, self.lattice.control_points)

            pos_vec1 = np.matmul(vertex.b_vec1, self.lattice.control_points)
            pos_vec2 = np.matmul(vertex.b_vec2, self.lattice.control_points)
            n = np.cross(pos_vec1-pos, pos_vec2-pos)
            n = n/np.linalg.norm(n)

            if index == 0:
                n_index = self.scene.dict_vertex_normal.get(str(index+1))
                ori_n = self.scene.normals[int(n_index)-1]
            n_index = self.scene.dict_vertex_normal.get(str(index+1))
            self.scene.normals[int(n_index)-1] = n

            vertex.position = pos
            screen_coord = Matrix().project(vertex.position[0],vertex.position[1],vertex.position[2],self.mv_mat,self.proj,0,0, self.window_width,self.window_height)
            self.scene.screen_coord[index] = screen_coord

        self.scene.kdtree = spatial.KDTree(self.scene.screen_coord[:,:2])
        self.scene.update_object(False)
        m = list(self.scene.objects.values())[0]
        self.mesh.vertices = m.vertices
        self.update_glsl(True)

    def on_touch_up(self,touch):
        if self.moving:
            self.moving = False
            if self.mode == 'display':
                self.set_lattice_screen_coord()
                if self.collide_point(touch.pos[0], touch.pos[1]):
                    self.p1 = self.trackball.get_hemisphere_coord(touch.pos[0], touch.pos[1])
            if self.mode == 'select':
                min_x = min([self.start[0],touch.pos[0]])
                min_y = min([self.start[1],touch.pos[1]])
                max_x = max([self.start[0],touch.pos[0]])
                max_y = max([self.start[1],touch.pos[1]])
                for lcp in self.lattice.vertices:
                    if lcp.screen_coord[0] > min_x and lcp.screen_coord[0] < max_x and lcp.screen_coord[1] > min_y and lcp.screen_coord[1] < max_y:
                        lcp.toggle_selection()
                        if lcp.selected:
                            self.lattice_selected += 1
                        else:
                            self.lattice_selected -= 1
                        self.update_lattice_mesh(lcp.index)
            if self.mode == 'edit':
                if self.moved:
                    if self.lattice_selected:
                        self.update_object_vertices()
                    self.moved == False
            if self.mode == 'direct':
                if self.moved and self.selected_mesh_vertex != None:
                    self.undo_stack.append(np.copy(self.lattice.control_points))
                    self.redo_stack.clear()
                    delta_p = np.array(touch.pos)-np.array(self.start)
                    point = self.transform_point(self.mv_arr,self.selected_mesh_vertex.position[0],self.selected_mesh_vertex.position[1],self.selected_mesh_vertex.position[2],1.0)
                    point = self.transform_point(self.proj_arr,point[0],point[1],point[2],point[3])
                    x_clip = delta_p[0]*2/self.window_width
                    y_clip = delta_p[1]*2/self.window_height
                    x_clip = x_clip * point[3]
                    y_clip = y_clip * point[3]
                    point = self.transform_point(self.proj_i_arr,x_clip,y_clip,point[2],point[3])
                    point = self.transform_point(self.mv_i_arr,point[0],point[1],point[2],point[3])
                    delta_p = np.array([point[:3]])
                    self.lattice.control_points = self.lattice.control_points+   np.matmul(self.selected_mesh_vertex.b_inv,delta_p)
                    self.moved == False
                    self.start = touch.pos
                    for lcp in self.lattice.vertices:
                        lcp.position = self.lattice.control_points[lcp.index]
                        self.update_lattice_mesh(lcp.index)
                    self.update_object_vertices()
            if self.mode == 'bend':
                end_index = np.abs(self.plane_y - touch.pos[1]).argmin()
                end_v_index = np.abs(self.plane_x - touch.pos[0]).argmin()
                if len(self.moving_path) > 10:
                    diff = np.array(self.moving_path[9])-np.array(self.start)
                    self.undo_stack.append(np.copy(self.lattice.control_points))
                    self.redo_stack.clear()
                    if abs(diff [1]) > abs(diff[0]):
                        stepsize = int(len(self.moving_path)/(abs(end_index-self.nearest_plane_index)+1))
                        for step, plane in enumerate(self.lattice.planes[self.nearest_plane_index+1:end_index+1]):
                            segment_endpoint = self.moving_path[(step+1)*stepsize]
                            displacement = np.array(segment_endpoint) - np.array(self.start)
                            angle = np.arctan(displacement[0]/displacement[1])
                            rotation = Matrix().rotate(angle,0.0,0.0,1.0).get()
                            for lcp in plane:
                                pos = lcp.position
                                new_pos = self.transform_point(rotation, pos[0],pos[1],pos[2],1.0)
                                new_pos = np.array(new_pos[:-1])
                                point = self.transform_point(self.mv_arr,new_pos[0],new_pos[1],self.plane_z,1.0)
                                point = self.transform_point(self.proj_arr,point[0],point[1],point[2],point[3])
                                x_clip = diff[0]*2/self.window_width
                                y_clip = diff[1]*2/self.window_height
                                x_clip = x_clip * point[3] + point[0]
                                y_clip = y_clip * point[3] + point[1]
                                new_pos[0] += displacement[0]
                                new_pos[1] += displacement[1]
                                point = self.transform_point(self.proj_i_arr,x_clip,y_clip,point[2],point[3])
                                point = self.transform_point(self.mv_i_arr,point[0],point[1],point[2],point[3])
                                lcp.position = [point[0],point[1],pos[2]]
                                self.lattice.control_points[lcp.index] = lcp.position
                                #print(pos,new_pos)
                                self.update_lattice_mesh(lcp.index)
                            #print(np.degrees(angle))
                    else:
                        stepsize = int(len(self.moving_path)/(abs(end_v_index-self.nearest_v_plane_index)+1))
                        for step, plane in enumerate(self.lattice.v_planes[self.nearest_v_plane_index+1:end_v_index+1]):
                            segment_endpoint = self.moving_path[(step+1)*stepsize]
                            displacement = np.array(segment_endpoint) - np.array(self.start)
                            angle = -np.arctan(displacement[1]/displacement[0])
                            rotation = Matrix().rotate(angle,0.0,0.0,1.0).get()
                            for lcp in plane:
                                pos = lcp.position
                                new_pos = self.transform_point(rotation, pos[0],pos[1],pos[2],1.0)
                                new_pos = np.array(new_pos[:-1])
                                point = self.transform_point(self.mv_arr,new_pos[0],new_pos[1],self.plane_z,1.0)
                                point = self.transform_point(self.proj_arr,point[0],point[1],point[2],point[3])
                                x_clip = diff[0]*2/self.window_width
                                y_clip = diff[1]*2/self.window_height
                                x_clip = x_clip * point[3] + point[0]
                                y_clip = y_clip * point[3] + point[1]
                                new_pos[0] += displacement[0]
                                new_pos[1] += displacement[1]
                                point = self.transform_point(self.proj_i_arr,x_clip,y_clip,point[2],point[3])
                                point = self.transform_point(self.mv_i_arr,point[0],point[1],point[2],point[3])
                                lcp.position = [point[0],point[1],pos[2]]
                                self.lattice.control_points[lcp.index] = lcp.position
                                #print(pos,new_pos)
                                self.update_lattice_mesh(lcp.index)
                    self.update_object_vertices()
            if self.mode == 'twist':
                end_index = np.abs(self.plane_y - touch.pos[1]).argmin()
                end_v_index = np.abs(self.plane_x - touch.pos[0]).argmin()
                if len(self.moving_path) > 10:
                    self.undo_stack.append(np.copy(self.lattice.control_points))
                    self.redo_stack.clear()
                    diff = np.array(self.moving_path[6])-np.array(self.start)
                    displacement = np.array(touch.pos) - np.array(self.start)
                    if abs(diff[0]) > abs(diff[1]):
                        theta = np.arctan(displacement[1]/displacement[0])/2
                        d_theta = theta*2/(abs(self.nearest_plane_index-end_index)+1)
                        print(theta,d_theta)
                        for plane in self.lattice.planes[:self.nearest_plane_index]:
                            rotation = Matrix().rotate(theta,0.0,1.0,0.0).get()
                            for lcp in plane:
                                pos = lcp.position
                                new_pos = self.transform_point(rotation, pos[0],pos[1],pos[2],1.0)
                                new_pos = np.array(new_pos[:-1])
                                lcp.position = [new_pos[0],new_pos[1],new_pos[2]]
                                self.lattice.control_points[lcp.index] = lcp.position
                                self.update_lattice_mesh(lcp.index)
                        for step, plane in enumerate(self.lattice.planes[self.nearest_plane_index:end_index+1]):
                            angle = theta-d_theta*step
                            rotation = Matrix().rotate(angle,0.0,1.0,0.0).get()
                            for lcp in plane:
                                pos = lcp.position
                                new_pos = self.transform_point(rotation, pos[0],pos[1],pos[2],1.0)
                                new_pos = np.array(new_pos[:-1])
                                lcp.position = [new_pos[0],new_pos[1],new_pos[2]]
                                self.lattice.control_points[lcp.index] = lcp.position
                                #print(pos,new_pos)
                                self.update_lattice_mesh(lcp.index)
                        for plane in self.lattice.planes[end_index+1:]:
                            rotation = Matrix().rotate(-theta,0.0,1.0,0.0).get()
                            for lcp in plane:
                                pos = lcp.position
                                new_pos = self.transform_point(rotation, pos[0],pos[1],pos[2],1.0)
                                new_pos = np.array(new_pos[:-1])
                                lcp.position = [new_pos[0],new_pos[1],new_pos[2]]
                                self.lattice.control_points[lcp.index] = lcp.position
                                self.update_lattice_mesh(lcp.index)
                    else:
                        theta = np.arctan(displacement[0]/displacement[1])/2
                        d_theta = theta*2/(abs(self.nearest_plane_index-end_index)+1)
                        print(theta,d_theta)
                        for plane in self.lattice.v_planes[:self.nearest_v_plane_index]:
                            rotation = Matrix().rotate(theta,1.0,0.0,0.0).get()
                            for lcp in plane:
                                pos = lcp.position
                                new_pos = self.transform_point(rotation, pos[0],pos[1],pos[2],1.0)
                                new_pos = np.array(new_pos[:-1])
                                lcp.position = [new_pos[0],new_pos[1],new_pos[2]]
                                self.lattice.control_points[lcp.index] = lcp.position
                                self.update_lattice_mesh(lcp.index)
                        for step, plane in enumerate(self.lattice.v_planes[self.nearest_v_plane_index:end_v_index+1]):
                            angle = theta-d_theta*step
                            rotation = Matrix().rotate(angle,1.0,0.0,0.0).get()
                            for lcp in plane:
                                pos = lcp.position
                                new_pos = self.transform_point(rotation, pos[0],pos[1],pos[2],1.0)
                                new_pos = np.array(new_pos[:-1])
                                lcp.position = [new_pos[0],new_pos[1],new_pos[2]]
                                self.lattice.control_points[lcp.index] = lcp.position
                                #print(pos,new_pos)
                                self.update_lattice_mesh(lcp.index)
                        for plane in self.lattice.planes[end_v_index+1:]:
                            rotation = Matrix().rotate(-theta,1.0,0.0,0.0).get()
                            for lcp in plane:
                                pos = lcp.position
                                new_pos = self.transform_point(rotation, pos[0],pos[1],pos[2],1.0)
                                new_pos = np.array(new_pos[:-1])
                                lcp.position = [new_pos[0],new_pos[1],new_pos[2]]
                                self.lattice.control_points[lcp.index] = lcp.position
                                self.update_lattice_mesh(lcp.index)
                    self.update_object_vertices()

class Menu(GridLayout):
    def __init__(self, **kwargs):
        super(Menu, self).__init__(**kwargs)

class FileChoosePopup(Popup):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class FileSavePopup(Popup):
    export = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def update_path(self, selection):
        self.ids.filesave_path.text=str(selection)[2:-2]

class FFDeformer(BoxLayout):
    def __init__(self, **kwargs):
        super(FFDeformer, self).__init__(**kwargs)
        self.ids.menu.ids.check_view.bind(active=self.change_mode_display)
        self.ids.menu.ids.check_select.bind(active=self.change_mode_select)
        self.ids.menu.ids.check_edit.bind(active=self.change_mode_edit)
        self.ids.menu.ids.check_direct.bind(active=self.change_mode_direct)
        self.ids.menu.ids.check_bend.bind(active=self.change_mode_bend)
        self.ids.menu.ids.check_twist.bind(active=self.change_mode_twist)
        self.ids.btn_cube.bind(on_release=partial(self.load_obj,'cube'))
        self.ids.btn_sphere.bind(on_release=partial(self.load_obj,'sphere'))
        self.ids.btn_cylinder.bind(on_release=partial(self.load_obj,'cylinder'))
        self.ids.btn_torus.bind(on_release=partial(self.load_obj,'torus'))
        self.ids.btn_from_file.bind(on_release=self.open_popup)
        self.ids.btn_save.bind(on_release=self.open_save)
        self.ids.fov_slider.bind(value=self.fov_value)
        self.ids.cam_front.bind(on_release=partial(self.update_cam,'front'))
        self.ids.cam_left.bind(on_release=partial(self.update_cam,'left'))
        self.ids.cam_top.bind(on_release=partial(self.update_cam,'top'))
        self.ids.cam_back.bind(on_release=partial(self.update_cam,'back'))
        self.ids.cam_right.bind(on_release=partial(self.update_cam,'right'))
        self.ids.cam_bottom.bind(on_release=partial(self.update_cam,'bottom'))
        self.ids.x_cp_slider.bind(value=self.x_cp_value)
        self.ids.y_cp_slider.bind(value=self.y_cp_value)
        self.ids.z_cp_slider.bind(value=self.z_cp_value)
        self.ids.btn_reset.bind(on_release=self.reset)
        self.ids.btn_undo.bind(on_release=self.undo)
        self.ids.btn_redo.bind(on_release=self.redo)
        self.ids.btn_cp_visible.bind(on_release=self.toggle_cp_visibility)
        self.ids.btn_obj_visible.bind(on_release=self.toggle_obj_visibility)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_up=self._on_keyboard_up)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_up=self._on_keyboard_up)
        self._keyboard = None

    def _on_keyboard_up(self, keyboard, keycode):
        if keycode[1] == 'v':
            self.ids.renderer.mode='display'
            self.ids.menu.ids.check_view.active = True
        elif keycode[1] == 'e':
            self.ids.renderer.mode='edit'
            self.ids.menu.ids.check_edit.active = True
        elif keycode[1] == 's':
            self.ids.renderer.mode='select'
            self.ids.menu.ids.check_select.active = True
        elif keycode[1] == 'd':
            self.ids.renderer.mode='direct'
            self.ids.menu.ids.check_direct.active = True
        elif keycode[1] == 'b':
            self.ids.renderer.mode='bend'
            self.ids.menu.ids.check_bend.active = True
        elif keycode[1] == 't':
            self.ids.renderer.mode='twist'
            self.ids.menu.ids.check_twist.active = True
        elif keycode[1] == 'r':
            self.ids.renderer.redo()
        elif keycode[1] == 'u':
            self.ids.renderer.undo()

    def redo(self,btn):
        self.ids.renderer.redo()

    def undo(self,btn):
        self.ids.renderer.undo()

    def toggle_cp_visibility(self, btn):
        self.ids.renderer.canvas.clear()
        if self.ids.renderer.lattice_visible:
            self.ids.renderer.lattice_visible = False
            self.ids.btn_cp_visible.text = 'Show Control Points'
        else:
            self.ids.renderer.lattice_visible = True
            self.ids.btn_cp_visible.text = 'Hide Control Points'
        self.ids.renderer.render_scene()

    def toggle_obj_visibility(self, btn):
        self.ids.renderer.canvas.clear()
        if self.ids.renderer.object_visible:
            self.ids.renderer.object_visible = False
            self.ids.btn_obj_visible.text = 'Show Object'
        else:
            self.ids.renderer.object_visible = True
            self.ids.btn_obj_visible.text = 'Hide Object'
        self.ids.renderer.render_scene()

    def reset(self, btn):
        self.ids.renderer.l = int(self.ids.x_cp_slider.value)
        self.ids.renderer.m = int(self.ids.y_cp_slider.value)
        self.ids.renderer.n = int(self.ids.z_cp_slider.value)
        self.ids.renderer.load_obj(self.ids.renderer.cur_obj)

    def x_cp_value(self, instance, value):
        value = int(value)
        self.ids.x_cp_val.text = str(value) + ' cps'

    def y_cp_value(self, instance, value):
        value = int(value)
        self.ids.y_cp_val.text = str(value) + ' cps'

    def z_cp_value(self, instance, value):
        value = int(value)
        self.ids.z_cp_val.text = str(value) + ' cps'

    def update_cam(self,dir, btn):
        self.ids.renderer.updata_obj_dir(dir)

    def fov_value(self, instance, value):
        value = int(value)
        self.ids.fov_val.text = str(value)
        self.ids.renderer.fov = math.tan(np.radians(value/2))*4
        self.ids.renderer.update_glsl(True)

    def load_obj(self,obj,btn):
        self.ids.dropdown.dismiss()
        self.ids.renderer.load_obj(obj)

    def open_popup(self,btn):
        self.ids.dropdown.dismiss()
        self.the_popup = FileChoosePopup(load=self.load)
        self.the_popup.open()

    def open_save(self,btn):
        if self.ids.renderer.scene != None:
            self.the_save = FileSavePopup(export=self.save_file)
            self.the_save.open()

    def load(self, selection):
        file_path = str(selection[0])
        self.the_popup.dismiss()
        self.ids.renderer.load_obj(file_path)

    def save_file(self, filename, path):
        objname = filename.split('.')[0]
        path = path +'/'+filename
        self.ids.renderer.scene.write_to_file(objname, path)
        self.the_save.dismiss()

    def change_mode_display(self,checkboxInstance, isActive):
        if isActive:
            self.ids.renderer.mode='display'

    def change_mode_select(self,checkboxInstance, isActive):
        if isActive:
            self.ids.renderer.mode='select'

    def change_mode_edit(self,checkboxInstance, isActive):
        if isActive:
            self.ids.renderer.mode='edit'

    def change_mode_direct(self,checkboxInstance, isActive):
        if isActive:
            self.ids.renderer.mode='direct'

    def change_mode_bend(self,checkboxInstance, isActive):
        if isActive:
            self.ids.renderer.mode='bend'

    def change_mode_twist(self,checkboxInstance, isActive):
        if isActive:
            self.ids.renderer.mode='twist'

class CustomDropDown(BoxLayout):
    pass

class FFDeformerApp(App):
    def build(self):
        f = FFDeformer()
        return f

if __name__ == '__main__':
    FFDeformerApp().run()
