# Running the demos

The demos are mostly JSON files containing physical scenario descriptions. Some of the demos have their own executables, callbacks, cameras, etc.

Make sure you build in release mode with the `-DCMAKE_BUILD_TYPE=release -DOPTIMIZATION_OPT=o2` flags at `cmake` configuration.

## Generic demos

All of the generic demos can be found in `demos/scenarios`. You can run the generic demos with standard runner using the following commands from the build folder (assuming you've compiled everything):

| Operating System | Terminal command | Notes |
| ---------------- | ---------------- | ----- |
| Linux/Mac | `./bin/api_viz $SLINGSHOT_PATH/demos/scenarios/scenario_name.json $SLINGSHOT_PATH/demos/api/standard_viz_config.json` | `$SLINGSHOT_PATH` is the location of the `slingshot` source files |
| Windows | `bin\api_viz %SLINGSHOT_PATH%\demos\scenarios\scenario_name.json %SLINGSHOT_PATH%\demos\api\standard_viz_config.json` | `%SLINGSHOT_PATH%` is the location of the `slingshot` source files |

## Demo example

You can play, pause, or step the simulation using the `Play` checkbox or the `Step one frame` button. There are also several keybinds for maneuvering and interacting with the scene.

| Action | Keys |
| ------ | ---- |
| Change camera position | WASD |
| Change camera orientation | Right-click and drag |
| Grab rigid body | Left-click and drag |

Only rigid bodies within 30m of the camera can be grabbed with the mouse.

<img src="https://slingshot-gifs.s3.amazonaws.com/blocks-smash-readme.gif">
