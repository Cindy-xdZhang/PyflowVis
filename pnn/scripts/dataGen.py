import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def generate_triangle_points(n_points, center, size):
    """
    Generate random points uniformly inside an equilateral triangle.
    :param n_points: number of points to generate
    :param center: (x, y) center of the triangle
    :param size: edge length of the triangle
    :return: ndarray of shape (n_points, 2)
    """
    # compute triangle vertices
    h = size * np.sqrt(3) / 2
    verts = np.array([
        [0, 2/3*h],
        [-size/2, -1/3*h],
        [ size/2, -1/3*h]
    ]) + center
    # sample barycentric coordinates
    u = np.random.rand(n_points, 1)
    v = np.random.rand(n_points, 1)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - (u + v)
    pts = u * verts[0] + v * verts[1] + w * verts[2]
    return pts

def generate_quad_points(n_points, center, size):
    """
    Generate random points uniformly inside a square (quadrilateral).
    :param n_points: number of points to generate
    :param center: (x, y) center of the square
    :param size: edge length of the square
    :return: ndarray of shape (n_points, 2)
    """
    half = size / 2
    x = np.random.uniform(-half, half, size=n_points) + center[0]
    y = np.random.uniform(-half, half, size=n_points) + center[1]
    return np.vstack((x, y)).T

def generate_circle_points(n_points, center, radius):
    """
    Generate random points uniformly inside a circle.
    :param n_points: number of points to generate
    :param center: (x, y) center of the circle
    :param radius: radius of the circle
    :return: ndarray of shape (n_points, 2)
    """
    r = radius * np.sqrt(np.random.rand(n_points))
    theta = 2 * np.pi * np.random.rand(n_points)
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return np.vstack((x, y)).T

def draw_registration_result(pts, title=""):
    """visualize multiple point clouds + axis"""
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    pts.append(axis)
    o3d.visualization.draw_geometries(pts, window_name=title)

if __name__ == "__main__":

    n_tri  = 400   # number of points in triangles
    n_quad = 400   # number of points in quadrilaterals
    n_circ = 400   # number of points in circles

    size_tri   = 0.5  # edge length of triangles
    size_quad  = 0.5  # edge length of squares
    radius_circ = 0.5 # radius of circles

    center_tri  = (0, 0)
    center_quad = (2, 0)
    center_circ = (1, 0)

    # get point clouds
    pts_tri  = generate_triangle_points(n_tri,  center_tri,  size_tri)
    pts_quad = generate_quad_points(n_quad, center_quad, size_quad)
    pts_circ = generate_circle_points(n_circ, center_circ, radius_circ)

    # 2d vis
    plt.figure(figsize=(8, 6))
    plt.scatter(pts_tri[:,0],  pts_tri[:,1],  s=10, label="Triangle",   alpha=0.6)
    plt.scatter(pts_quad[:,0], pts_quad[:,1], s=10, label="Quadrilateral",alpha=0.6)
    plt.scatter(pts_circ[:,0], pts_circ[:,1], s=10, label="Circle",       alpha=0.6)
    plt.axis("equal")
    plt.legend()
    plt.title("Random 2D Point Clouds in Different Shapes")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
    # 3d vis
    pcd_tri = o3d.geometry.PointCloud()
    pcd_quad = o3d.geometry.PointCloud()
    pcd_cric = o3d.geometry.PointCloud()
    
    # 2. pcd.points / Assign points
    ones = np.zeros((pts_tri.shape[0], 1))
    pts_tri_3d=np.concatenate([pts_tri,ones],axis=1)
    ones = np.zeros((pts_quad.shape[0], 1))
    pts_quad_3d=np.concatenate([pts_quad,ones],axis=1)
    ones = np.zeros((pts_circ.shape[0], 1))
    pts_circ_3d=np.concatenate([pts_circ,ones],axis=1)
    pcd_tri.points = o3d.utility.Vector3dVector(pts_tri_3d)
    pcd_quad.points = o3d.utility.Vector3dVector(pts_quad_3d)
    pcd_cric.points = o3d.utility.Vector3dVector(pts_circ_3d)
    # 
    pcd_tri.paint_uniform_color([0.0, 0.0, 0.0])# black 
    pcd_quad.paint_uniform_color([0.0, 1.0, 0.0])# green
    pcd_cric.paint_uniform_color([0.0, 0.0, 1.0])# blue
    draw_registration_result([pcd_tri, pcd_quad,pcd_cric], title="visulaize 3D scene(pcd_tri in balck, pcd_quad in green, pcd_cric in blue)")