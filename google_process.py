import os
import zipfile
import bpy
import math
import shutil

bpy.app.binary_path=os.path.join("blend","blender-5.0.1-linux-x64","blender")
src_dir="google"
files=[file for file in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,file))]
print(files)

'''for zf in files:
    print(f"processing {zf}...",end="")
    dest=zf.split(".")[0]
    dest_path=os.path.join(src_dir,dest)
    os.makedirs(dest_path,exist_ok=True)
    with zipfile.ZipFile(os.path.join(src_dir,zf), 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    print("done!")'''
    

OUT_DIR = "renders"
os.makedirs(OUT_DIR)

NUM_VIEWS = 5
RADIUS = 2.0
ELEVATION = math.radians(30)

IMAGE_RES = 256
ENGINE = "CYCLES"  # or "BLENDER_EEVEE"
    
for file in files:
    subfolders=os.listdir(os.path.join(src_dir,file))
    print(subfolders)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    shutil.copy(os.path.join(src_dir,file,"materials","texture.png"), os.path.join(src_dir,file,"materials","texture.png"))
    bpy.ops.wm.obj_import(filepath=os.path.join(src_dir,file,"meshes","model.obj"),)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    # -----------------------
    # NORMALIZE SCALE
    # -----------------------
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)

    max_dim = max(obj.dimensions)
    obj.scale /= max_dim

    bpy.ops.object.transform_apply(scale=True)

    # -----------------------
    # CAMERA
    # -----------------------
    cam_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(camera)

    bpy.context.scene.camera = camera

    # -----------------------
    # LIGHTING
    # -----------------------
    light_data = bpy.data.lights.new("KeyLight", type="AREA")
    light_data.energy = 1000
    light = bpy.data.objects.new("KeyLight", light_data)
    light.location = (2, 2, 2)
    bpy.context.collection.objects.link(light)

    # -----------------------
    # RENDER SETTINGS
    # -----------------------
    scene = bpy.context.scene
    scene.render.engine = ENGINE
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = IMAGE_RES
    scene.render.resolution_y = IMAGE_RES
    scene.render.resolution_percentage = 100

    if ENGINE == "CYCLES":
        scene.cycles.samples = 128
        scene.cycles.device = 'GPU' if bpy.context.preferences.addons['cycles'].preferences.compute_device_type != 'NONE' else 'CPU'

    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------
    # RENDER LOOP
    # -----------------------
    for i in range(NUM_VIEWS):
        angle = 2 * math.pi * i / NUM_VIEWS

        cam_x = RADIUS * math.cos(angle)
        cam_y = RADIUS * math.sin(angle)
        cam_z = RADIUS * math.sin(ELEVATION)

        camera.location = (cam_x, cam_y, cam_z)
        camera.rotation_euler = (
            math.radians(90) - ELEVATION,
            0,
            angle + math.pi / 2
        )

        scene.render.filepath = os.path.join(OUT_DIR, f"view_{i:03d}.png")
        bpy.ops.render.render(write_still=True)

    print("Done")
    break
        