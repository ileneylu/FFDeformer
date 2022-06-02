import numpy as np

class Object():

    def __init__(self, name=''):
        self.name = name
        self.faces = []
        self.vertices = []
        self.face_vertex_uvs = [[]]

    def compute_vertex_normal(self):
        pass

class Box(Object):
    _cube_vertices = np.array([(-1, 1, -1), (1, 1, -1),
                      (1, -1, -1), (-1, -1, -1),
                      (-1, 1, 1), (1, 1, 1),
                      (1, -1, 1), (-1, -1, 1),
                      ])

    _cube_faces = np.array([(0, 1, 2), (0, 2, 3), (3, 2, 6),
                   (3, 6, 7), (7, 6, 5), (7, 5, 4),
                   (4, 5, 1), (4, 1, 0), (4, 0, 3),
                   (7, 4, 3), (5, 1, 2), (6, 5, 2)
                   ])

    _cube_normals = np.array([(0, 0, 1), (-1, 0, 0), (0, 0, -1),
                     (1, 0, 0), (0, 1, 0), (0, -1, 0)
                     ])

     def __init__(self, width, height, depth, **kw):
        name = kw.pop('name', '')
        super(Box, self).__init__(name)
        self.w = width
        self.h = height
        self.d = depth
