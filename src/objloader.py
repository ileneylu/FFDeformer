import numpy as np

class MeshData(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.vertex_format = [
            (b'v_pos', 3, 'float'),
            (b'v_color', 3, 'float'),
            (b'v_normal', 3, 'float')]
        self.vertices = []
        self.indices = []

    def calculate_normals(self):
        for i in range(int(len(self.indices) / (3))):
            fi = i * 3
            v1i = self.indices[fi]
            v2i = self.indices[fi + 1]
            v3i = self.indices[fi + 2]

            vs = self.vertices
            p1 = [vs[v1i + c] for c in range(3)]
            p2 = [vs[v2i + c] for c in range(3)]
            p3 = [vs[v3i + c] for c in range(3)]

            u, v = [0, 0, 0], [0, 0, 0]
            for j in range(3):
                v[j] = p2[j] - p1[j]
                u[j] = p3[j] - p1[j]

            n = [0, 0, 0]
            n[0] = u[1] * v[2] - u[2] * v[1]
            n[1] = u[2] * v[0] - u[0] * v[2]
            n[2] = u[0] * v[1] - u[1] * v[0]

            for k in range(3):
                self.vertices[v1i + 3 + k] = n[k]
                self.vertices[v2i + 3 + k] = n[k]
                self.vertices[v3i + 3 + k] = n[k]

class Vertex():
    def __init__(self, pos):
        self.position = pos

class ObjFile:
    def write_to_file(self, name, path):
        with open(path, 'w') as file:
            file.write('o ' +str(name) +'\n')
            for vertex in self.vertices:
                file.write('v ' +str(vertex.position[0]) + ' ' + str(vertex.position[1]) + ' ' + str(vertex.position[2]) + '\n')
            for normal in self.normals:
                file.write('vn ' +str(normal[0]) + ' ' + str(normal[1]) + ' ' + str(normal[2]) + '\n')
            file.write('s off\n')
            for face in self.faces:
                file.write('f')
                verts = face[0]
                for i in range(3):
                    n_index = self.dict_vertex_normal.get(str(verts[i]))
                    file.write(' ' + str(verts[i])+'//'+str(n_index))
                file.write('\n')

    def init_vertex_info(self, ori, lattice_size, knot_size):
        for vertex in self.vertices:
            vertex.local_coord = np.array(vertex.position) - np.array(ori)
            vertex.local_coord =  vertex.local_coord/np.array(lattice_size)*np.array(knot_size)
            vertex.segment = np.floor(vertex.local_coord)


    def finish_object(self):
        if self._current_object is None:
            return

        mesh = MeshData()
        idx = 0
        for f in self.faces:
            verts = f[0]
            norms = f[1]
            for i in range(3):
                # get vertex components
                v = self.vertices[verts[i] - 1].position
                c = [0.27,0.57,0.52]

                # get normal components
                n_index = self.dict_vertex_normal.get(str(verts[i]))
                n = self.normals[int(n_index)-1]

                data = [v[0], v[1], v[2], c[0], c[1], c[2], n[0], n[1], n[2]]
                mesh.vertices.extend(data)

            tri = [idx, idx + 1, idx + 2]
            mesh.indices.extend(tri)
            idx += 3

        self.objects[self._current_object] = mesh


    def update_object(self,calculate_normals):
        if self._current_object is None:
            return

        mesh = MeshData()
        idx = 0
        for f in self.faces:
            verts = f[0]
            norms = f[1]
                # get vertex components
            v1 = np.array(self.vertices[verts[0] - 1].position)
            v2 = np.array(self.vertices[verts[1] - 1].position)
            v3 = np.array(self.vertices[verts[2] - 1].position)
            c = [0.27,0.57,0.52]

            if calculate_normals:
                u = v2-v1
                v = v3-v1
                n = np.cross(u,v)
                n1 = n/np.linalg.norm(n)
                n2 = n1
                n3 = n1
            else:
                n_index_1 = self.dict_vertex_normal.get(str(verts[0]))
                n_index_2 = self.dict_vertex_normal.get(str(verts[1]))
                n_index_3 = self.dict_vertex_normal.get(str(verts[2]))
                n1 = self.normals[int(n_index_1) - 1]
                n2 = self.normals[int(n_index_2) - 1]
                n3 = self.normals[int(n_index_3) - 1]
            #print(n,self.normals[norms[0] - 1])

            data1 = [v1[0], v1[1], v1[2], c[0], c[1], c[2], n1[0], n1[1], n1[2]]
            data2 = [v2[0], v2[1], v2[2], c[0], c[1], c[2], n2[0], n2[1], n2[2]]
            data3 = [v3[0], v3[1], v3[2], c[0], c[1], c[2], n3[0], n3[1], n3[2]]
            mesh.vertices.extend(data1)
            mesh.vertices.extend(data2)
            mesh.vertices.extend(data3)

            tri = [idx, idx + 1, idx + 2]
            mesh.indices.extend(tri)
            idx += 3
        self.objects[self._current_object] = mesh

    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.objects = {}
        self.vertices = []
        self.normals = []
        self.normals_set = []
        self.faces = []

        self._current_object = None

        material = None
        self.dict_vertex_normal = {}
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            if line.startswith('s'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'o':
                self.finish_object()
                self._current_object = values[1]
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(Vertex(v))
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals_set.append(v)
            elif values[0] == 'f':
                face = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')

                    query_result = self.dict_vertex_normal.get(w[0])
                    if query_result != None:
                        if query_result != w[2]:
                            v_position = self.vertices[int(w[0])-1].position
                            self.vertices.append(Vertex(v_position))
                            idx = len(self.vertices)
                            face.append(idx)
                            n = self.normals_set[int(w[2])-1]
                            self.normals.append(n)
                            n_index = len(self.normals)
                            norms.append(n_index)
                            self.dict_vertex_normal[str(idx)] = str(n_index)
                        else:
                            face.append(int(w[0]))
                            norms.append(int(w[2]))
                    else:
                        face.append(int(w[0]))
                        n = self.normals_set[int(w[2])-1]
                        self.normals.append(n)
                        n_index = len(self.normals)
                        norms.append(n_index)
                        self.dict_vertex_normal[w[0]] = str(n_index)
                self.faces.append((face, norms))
        self.num_vertex = len(self.vertices)
        self.num_normal = len(self.normals)
        self.finish_object()

def MTL(filename):
    contents = {}
    mtl = None
    return
    for line in open(filename, "r"):
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError("mtl file doesn't start with newmtl stmt")
        mtl[values[0]] = values[1:]
    return contents
