import bpy
import os
import math

def render_obj(
    out_dir,
    engine,
    image_res,
    num_views,
    radius,
    elevation,
    num_views_z,
    count
):
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
    '''light_data = bpy.data.lights.new("KeyLight", type="AREA")
    light_data.energy = 1000
    light = bpy.data.objects.new("KeyLight", light_data)
    light.location = (2, 2, 2)
    bpy.context.collection.objects.link(light)'''
    
    
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    nodes.clear()

    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # RGB + Alpha
    bg.inputs["Strength"].default_value = 0.5              # Ambient intensity

    out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    # -----------------------
    # RENDER SETTINGS
    # -----------------------
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = image_res
    scene.render.resolution_y = image_res
    scene.render.resolution_percentage = 100

    if engine == "CYCLES":
        scene.cycles.samples = 128
        scene.cycles.device = 'GPU' if bpy.context.preferences.addons['cycles'].preferences.compute_device_type != 'NONE' else 'CPU'

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------
    # RENDER LOOP
    # -----------------------
    for i in range(num_views):
        for z in range(num_views_z):
            angle = 2 * math.pi * i / num_views
            z_angle=math.pi*z/num_views_z

            cam_x = radius * math.cos(angle)
            cam_y = radius * math.sin(angle)
            cam_z = radius * math.sin(z_angle)

            camera.location = (cam_x, cam_y, cam_z)
            camera.rotation_euler = (
                math.radians(90) - z_angle,
                0,
                angle + math.pi / 2
            )

            scene.render.filepath = os.path.join(out_dir, f"view_{count:03d}.png")
            bpy.ops.render.render(write_still=True)
        
            count+=1
    return count