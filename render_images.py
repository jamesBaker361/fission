import bpy
import os
import math
import random

def render_obj(
    out_dir,
    engine,
    image_res,
    num_views,
    radius,
    num_views_z,
    num_views_random,
    csv_path,
    category,
    instance,
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
            azimuth = 2 * math.pi * i / num_views
            polar = math.pi * (z + 0.5) / num_views_z  # avoid poles

            cam_x = radius * math.sin(polar) * math.cos(azimuth)
            cam_y = radius * math.sin(polar) * math.sin(azimuth)
            cam_z = radius * math.cos(polar)

            camera.location = (cam_x, cam_y, cam_z)

            # Look at origin
            direction = obj.location - camera.location
            camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
            path=os.path.join(
                out_dir, f"view_{count:03d}.png"
            )
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)

            count += 1
            rotation=f"{radius * math.sin(polar) * math.cos(azimuth)}_{radius * math.sin(polar) * math.sin(azimuth)}_{radius * math.cos(polar)}"
            location=f"{cam_x}_{cam_y}_{cam_z}"
            
            with open(csv_path,"a") as csv_file:
                csv_file.write(f"{path},{category},{instance},{location},{rotation}\n")
                print(f"wrote {path},{category},{instance},{location},{rotation} to {csv_path}")
    for r in range(num_views_random):
        azimuth = 2 * math.pi * random.random()
        polar = math.pi * random.random() # (z + 0.5) / num_views_z  # avoid poles

        cam_x = radius * math.sin(polar) * math.cos(azimuth)
        cam_y = radius * math.sin(polar) * math.sin(azimuth)
        cam_z = radius * math.cos(polar)

        camera.location = (cam_x, cam_y, cam_z)

        # Look at origin
        direction = obj.location - camera.location
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        path=os.path.join(
            out_dir, f"{category}_{instance}_view_{count:03d}.png"
        )
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)

        count += 1
        rotation=f"{radius * math.sin(polar) * math.cos(azimuth)}_{radius * math.sin(polar) * math.sin(azimuth)}_{radius * math.cos(polar)}"
        location=f"{cam_x}_{cam_y}_{cam_z}"
        
        with open(csv_path,"a") as csv_file:
            csv_file.write(f"{path},{category},{instance},{location},{rotation}\n")
            print(f"wrote {path},{category},{instance},{location},{rotation} to {csv_path}")
        
    return count