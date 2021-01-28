[![Build-Run-Windows](https://github.com/therealjtgill/slingshot/actions/workflows/build-windows.yml/badge.svg)](https://github.com/therealjtgill/slingshot/actions/workflows/build-windows.yml) [![Build-Run-Ubuntu](https://github.com/therealjtgill/slingshot/actions/workflows/build-ubuntu.yml/badge.svg?branch=)](https://github.com/therealjtgill/slingshot/actions/workflows/build-ubuntu.yml) [![Nightly-Run-Ubuntu](https://github.com/therealjtgill/slingshot/actions/workflows/nightly_scenarios.yml/badge.svg)](https://github.com/therealjtgill/slingshot/actions/workflows/nightly_scenarios.yml)

![slingshot](https://slingshot-gifs.s3.amazonaws.com/slingshot-readme.png)

# Description

`slingshot` is a constraint-based physics engine for 3D rigid body dynamics. It's driven by a custom ECS (T-RECS) and runs on Linux, Mac, and Windows. Scenarios can be created in an editor and saved to JSON, or scenarios can be created via access to a high-level API. The engine can be executed with an OpenGL-backed renderer or headlessly.

|  |  |
|:---:|:---:|
|<img src="https://slingshot-gifs.s3.amazonaws.com/ld52-small-readme.gif" width=300>| <img src="https://slingshot-gifs.s3.amazonaws.com/piston_table_latest_readme.gif" width=300> |
| <img src="https://slingshot-gifs.s3.amazonaws.com/ragdoll-sandbox-readme.gif" width=300> | <img src="https://slingshot-gifs.s3.amazonaws.com/blocks-smash-readme.gif" width=300> |

# Run the demos

There are a few demo integration scenarios that verify the engine is working, these are completely controlled through scenario files. There are also demo UI's that test specific algorithms used in the engine, such as `quickhull`.

The following example commands run the ragdoll sandbox scenario shown at the top of the README.

<span style="color:magenta">See the markdown file in the `demos` folder for more details.</span>

## Requirements

- `gcc` >= 8.1
- `cmake` >= 3.16
- `opengl` >= 3.3
- `mingw` (Windows only)

## Building

### Windows

1. Pull the repo from github (assuming you're using a command-line git tool)

   ```posh
   git clone https://github.com/therealjtgill/slingshot.git
   ```

2. Initialize all of the submodules

   ```posh
   cd slingshot
   git submodule update --init
   ```

3. Run the `makeme.bat` script in the repo's main folder (I use git bash on Windows)

   ```posh
   makeme.bat -DBUILD_TESTS=0
   ```

4. Demos have to be executed from the build folder. Scenario files are in `slingshot\demos\scenarios` and can be passed to the default engine visualizer as command line arguments.

   #### Example

   ```posh
   cd build
   bin\api_viz.exe ..\demos\scenarios\ragdoll_sandbox.json ..\demos\api\standard_viz_config.json
   ```

   The default engine visualizer has to be executed from the build folder because of relative path requirements.

### Linux/Mac

1. Pull the repo from github

   ```console
   user@pc:~$ git clone https://github.com/therealjtgill/slingshot.git
   ```

2. Initialize all of the submodules

   ```console
   user@pc:~/slingshot$ git submodule update --init
   ```

3. Run the `makeme.sh` script in the repo's main folder

   for debug

   ```console
   user@pc:~/slingshot$ makeme.sh -DBUILD_TESTS=0
   ```

   or release (speedyboi)

   ```console
   user@pc:~/slingshot$ makeme.sh -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=0 -DOPTIMIZATION_OPT=O2
   ```

4. Demos have to be executed from the build folder. Scenario files are in `slingshot/demos/scenarios` and can be passed to the default engine visualizer as command line arguments.

   #### Example

   ```console
   user@pc:~/slingshot$ cd build
   user@pc:~/slingshot/build$ ./bin/api_viz ../demos/scenarios/ragdoll_sandbox.json ../demos/api/standard_viz_config.json
   ```

   The default engine visualizer has to be executed from the build folder because of relative path requirements.

# Editor


### Editor-based scenario generation

![readme_editor_example](https://slingshot-gifs.s3.amazonaws.com/scenario-editor-readme.gif)

### Scenario created in the editor running in the engine

![readme_editor_scenario](https://slingshot-gifs.s3.amazonaws.com/editor-scenario-run-readme.gif)


## Run the editor

1. Follow the build instructions for your operating system

2. From the `build` folder, run the `editor` executable

   Linux
   ```console
   user@pc:~/slingshot/build$ ./bin/editor
   ```

   Windows
   ```posh
   bin\editor.exe
   ```

   The editor must be run from the `build` folder because of relative path requirements.

## Features

| System         | Current     | Future |
|--------------|-----------|------------|
| Integration  | RK4       | -        |
| Broad Collision Detection | O(N^2)  | AABB BVH-2[^2] |
| Supported shapes | Cuboid, sphere, capsule, cylinder | plane, triangle, tetrahedron, convex mesh, height maps[^1], marching cubes[^1] |
| Narrow Collision Detection | GJK + SAT | - |
| Collision Feature Detection | EPA + SAT | - |
| Contact Manifold Generation | One-shot plane clipping with Sutherland-Hodgman  | -  |
| Constraint Solver | PGS (Projected Gauss-Seidel)  | Lemke[^1] or a blend of existing methods  |
| Equality Constraints | Revolute motors, gears, 1-dof translation, 1-dof rotation | Prismatic motors, contact-based gears |
| Inequality Constraints | Collision, translational friction, torsional friction | 1-dof rotation and translation limits, distance limits |
| Math library | Custom + GLM for rendering  | -  |
| ECS | T-RECS (custom bit-based ECS) | - |
| Test suite | catch2  | -  |
| GUI | Dear ImGui  | -  |
| Renderer | Custom OpenGL renderer  | Vulkan[^1]  |
| Scenario editor | Custom scenario editor  | -  |
| Data serialization | JSON  | -  |
| Parallelization | Single-threaded  | CUDA[^2] and Vulkan[^1]  |

[^1]: maybe

[^2]: in-progress


# Notes

I started working on this engine because I wanted a simulation tool that I had complete control over.  `slingshot` has been a learning experience for a ton of different topics - ECS architecture, rendering, differential geometry, constrained optimization, and more. I kept the feature set small because I wanted to iterate quickly on architecture while I developed algorithms, and supporting too many features off the bat seemed like the quickest way to build a Gordian Knot.

You might be asking:

      "Jon, why is there so little user interaction baked into this simulation?"
      
The reason is that I've been much more focused on developing robust algorithms than a user experience.

      "What could possibly be more important that the user experience, Jon, you nincompoop?"

ROBOTS. Robots are far more important than users. And they're not forcing me to say that, either. The primary goal for this physics engine is to build a fast, parallelizable simulation that can be used to evaluate control algorithms for robotic systems. That's how this whole thing started, actually.

      "Oh?"

Yep. I started writing the engine because I wanted to make better flight controllers and state estimators for my drone. But I'm also vain (insane?) enough to prefer hand-making software over using off-the-shelf components.

      "But... it sounds like there are way more features here than you'd need to simulate a drone?"

CORRECT. I learned about and fell in love with constraint-based dynamics shortly after I made the first commits. I decided that if this physics engine was going to handle quadcopters then it should handle legged and articulated robots, too.

      "Yeah ok, that's all fine, I guess. But this doesn't work with any existing robotics software. There's no RoS integration or plug-ins for Unreal or Unity."

...That wasn't a question, but you're right.

      "So what are you going to do about that?"

I don't know. Probably ask for help at some point. You've probably noticed that those kinds of integrations aren't on the roadmap, but it's not because I've forgotten about them; I'm just prioritizing core features over API integration. I figure it makes more sense to make the engine do something interesting first and then work on integrating it into other platforms.

      "Well that doesn't exactly help me *right now*, but it makes sense. You can still use this engine to make games and stuff, right?"

Absolutely! I submitted a game for Ludum Dare 52 built with this engine. It was more of a tech demo than a "good game" (and my ratings back that up), but helped make the effort feel "real" - other people ran my code on their machines. That experience was a diversion from the roadmap, but it helped me find and fix a ton of features.

However, at its core, this tool is meant for simulation of robots. Implementing sensors and controller plug-ins will take priority over features for game physics.

# More demos

|   |   |
|---|---|
| <img src="https://slingshot-gifs.s3.amazonaws.com/capsules-small-readme.gif" width=300> | <img src="https://slingshot-gifs.s3.amazonaws.com/keva-small-readme.gif" width=300> |
