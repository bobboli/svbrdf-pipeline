import sys
import os
import bpy
from bpy import data as D
from bpy import context as C
from mathutils import *
from math import *
import json

###########################
# Modify here:            #
configFilePath = r'F:\svbrdf-pipeline\photos\green_towel_low\out\config.json'
###########################
# Usage:
# >>> script = bpy.data.texts['SetupScene.py']
# >>> exec(script.as_string())

if __name__ == '__main__':
    configFilePath = os.path.realpath(configFilePath)
    rootPath = os.path.dirname(configFilePath)
    config = None
    try:
        with open(configFilePath, 'r') as f:
            config = json.load(f)
    except Exception:
        raise Exception('Invalid config file!')


    map_files = config['map_files']
    map_list = [
        'basecolor',
        'metallic',
        'specular',
        'speculartint',
        'roughness',
        'anisotropic',
        'normal',
        'tangent'
    ]

    # Modify camera and plane object
    scene = bpy.data.scenes['Scene']
    scene.render.resolution_x = config['size_x']
    scene.render.resolution_y = config['size_y']
    scene.cycles.film_exposure = config['exposure_time']

    camera = bpy.data.cameras['Camera']
    # Note: Blender uses width-based EFL, while industry standard features diagonal-based EFL.
    # They are a little different. See https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
    camera.lens = config['efl_35mm'] * (45.0 / 43.27)
    cameraObj = bpy.data.objects['Camera']
    cameraObj.location = Vector((0, 0, 1))
    cameraObj.rotation_euler = Euler((0, 0, 0))

    planeObj = bpy.data.objects['Plane']
    planeObj.location = Vector((0, 0, 0))
    planeObj.rotation_euler = Euler((0, 0, 0))
    
    planeObj.dimensions = Vector((config['delta_xy'] * config['size_x'], config['delta_xy'] * config['size_y'], 0))



    # Load new textures
    material = bpy.data.objects['Plane'].active_material
    for map_name in map_files:
        # Remove old texture
        if material.node_tree.nodes[map_name].image is not None:
            bpy.data.images.remove(material.node_tree.nodes[map_name].image)
        # Load new texture and bind to the node
        map = bpy.data.images.load(os.path.join(rootPath, map_files[map_name]))
        material.node_tree.nodes[map_name].image = map
        

   # material.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 
    
    
   # In Blender Cycles, strength of point light is its power, in Watt
    strength = max(config['light_intensity'])
    lamp = bpy.data.lights['Lamp']
    lamp.node_tree.nodes['Emission'].inputs['Strength'].default_value = strength * 4 * pi
    lamp.node_tree.nodes['Emission'].inputs['Color'].default_value[0] = config['light_intensity'][0] / strength
    lamp.node_tree.nodes['Emission'].inputs['Color'].default_value[1] = config['light_intensity'][1] / strength
    lamp.node_tree.nodes['Emission'].inputs['Color'].default_value[2] = config['light_intensity'][2] / strength

    lampObj = bpy.data.objects['Lamp']
    lampObj.location = Vector((config['light_position'][0], config['light_position'][1], config['light_position'][2]))
