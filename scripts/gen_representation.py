import numpy as np
import cv2 
import trimesh
import json

def to_homo(pts):
  '''
  @pts: (N,3 or 2) will homogeneliaze the last dimension
  '''
  assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
  homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
  return homo

def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0,255,0), linewidth=2):
  '''Revised from 6pack dataset/inference_dataset_nocs.py::projection
  @bbox: (2,3) min/max
  @line_color: RGB
  '''
  min_xyz = bbox.min(axis=0)
  xmin, ymin, zmin = min_xyz
  max_xyz = bbox.max(axis=0)
  xmax, ymax, zmax = max_xyz

  def draw_line3d(start,end,img):
    pts = np.stack((start,end),axis=0).reshape(-1,3)
    pts = (ob_in_cam@to_homo(pts).T).T[:,:3]   #(2,3)
    projected = (K@pts.T).T
    uv = np.round(projected[:,:2]/projected[:,2].reshape(-1,1)).astype(int)   #(2,2)
    img = cv2.line(img, uv[0].tolist(), uv[1].tolist(), color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)
    return img

  for y in [ymin,ymax]:
    for z in [zmin,zmax]:
      start = np.array([xmin,y,z])
      end = start+np.array([xmax-xmin,0,0])
      img = draw_line3d(start,end,img)

  for x in [xmin,xmax]:
    for z in [zmin,zmax]:
      start = np.array([x,ymin,z])
      end = start+np.array([0,ymax-ymin,0])
      img = draw_line3d(start,end,img)

  for x in [xmin,xmax]:
    for y in [ymin,ymax]:
      start = np.array([x,y,zmin])
      end = start+np.array([0,0,zmax-zmin])
      img = draw_line3d(start,end,img)

  return img

def project_3d_to_2d(pt,K,ob_in_cam):
  pt = pt.reshape(4,1)
  projected = K @ ((ob_in_cam@pt)[:3,:])
  projected = projected.reshape(-1)
  projected = projected/projected[2]
  return projected.reshape(-1)[:2].round().astype(int)

def draw_xyz_axis(color, ob_in_cam, scale=0.1, K=np.eye(3), thickness=3, transparency=0,is_input_rgb=False):
  '''
  @color: BGR
  '''
  if is_input_rgb:
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
  xx = np.array([1,0,0,1]).astype(float)
  yy = np.array([0,1,0,1]).astype(float)
  zz = np.array([0,0,1,1]).astype(float)
  xx[:3] = xx[:3]*scale
  yy[:3] = yy[:3]*scale
  zz[:3] = zz[:3]*scale
  origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
  xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
  yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
  zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
  line_type = cv2.LINE_AA
  arrow_len = 0
  tmp = color.copy()
  tmp1 = tmp.copy()
  tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp1 = tmp.copy()
  tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp1 = tmp.copy()
  tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp = tmp.astype(np.uint8)
  if is_input_rgb:
    tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

  return tmp

def read_json(path):
    with open(path, 'r') as file:
        return json.load(file) 
    
def get_transformation(data):
    transformations = []
    
    for d in data:
        R = np.array(d["R"])
        t = np.array(d["t"])
        
        T = np.hstack((R, t.reshape(-1, 1)))  # Concatenar R y t
        T = np.vstack((T, np.array([0, 0, 0, 1])))  # Añadir la fila [0, 0, 0, 1]
        transformations.append(T)
    return transformations

def get_K(data):
  return np.array([[data["fx"], 0, data["cx"]], [0, data["fy"], data["cy"]], [0,0,1]])

def calculate_bbox_3d(path):
    mesh = trimesh.load(path)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    return to_origin, np.stack([-extents/2, extents/2], axis=0).reshape(2,3)


import sys
to_origin, bbox = calculate_bbox_3d(f"/mnt/foundpose/datasets/{sys.argv[1]}/lmo/models/obj_000001.ply")

img = cv2.imread(f"/mnt/foundpose/datasets/{sys.argv[1]}/lmo/test/000001/rgb/000001.png")
K = get_K(read_json(f"/mnt/foundpose/datasets/{sys.argv[1]}/lmo/camera.json"))
transformations  =get_transformation(read_json(f"/mnt/foundpose/datasets/{sys.argv[1]}/inference/lmo_v1/1/estimated-poses.json"))

for T in transformations:
    center_pose = T@np.linalg.inv(to_origin)
    img= draw_posed_3d_box(K, img, center_pose, bbox)
    img = draw_xyz_axis(img, center_pose, 0.1, K, thickness=3, transparency=0, is_input_rgb=True)
cv2.imwrite("./results/imagen.png", img)