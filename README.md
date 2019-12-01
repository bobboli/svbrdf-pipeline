# svbrdf-pipeline
Generating svBRDF textures from photos taken with mobile phone. [[paper]](svbrdf-pipeline.pdf)

## Usage

### System Requirements

The pipeline is implemented in Python 3 with the following packages as additional dependencies:

- Numpy
- Scipy
- matplotlib
- sklearn
- cv2
- exifread

Anaconda distribution is recommended for simplifying the setup of the environment.

### Input Images Preparation

The pipeline requires a minimum of two input photos: one under the ambient illumination and one under the flash light. Then place them under the /photos and create a JSON file describing the input images. There is one example photos/green_towel_low, which you can refer to. You should preserve the [EXIF](https://en.wikipedia.org/wiki/Exif) of the images, since information like exposure time and focal length is required.

For each camera, you also need to shoot a photo of a 18% gray card to decide the intensity of the flash light, and a series of photos of a generated color palette (./photos/colorcard.jpg) to recover the camera's response curve. These are described as calibration images. Include them in the [list_json_file] (like ./photos/list_low.json) as well as all the folders of the materials you would like to process.

### Run the Process Script

Prepare all the required input photos as above, and run

```bash
python ./scripts/fit_all.py ./photos/[list_json_file]
```

For the example contained in the repository

```
python ./scripts/fit_all.py ./photos/list_low.json
```

And svBRDF textures will be generated in ./photos/[material_name]/out/

### Live Preview in Blender

A Python script ./blender/SetupScene.py can be used to quickly import all the textures into a 3D scene ./blender/material.blend. Tested on [Blender 2.80](https://www.blender.org/download/releases/2-80/).