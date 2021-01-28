# Types of tests

`slingshot` uses two types of tests: `exec_tests` and `viz_tests`. 

`exec_tests` (executable tests) are automated tests that automatically verify that algorithms work according to randomized, procedural, or hand-coded inputs.

`viz_tests` (visual tests) are not automated. These are executables with small GUI elements that let the user visually verify that an algorithm is working. Some algorithms are difficult to write procedural tests for, and hand-coded test input might give good feelings, but only cover a sliver of possible test cases.

## Executable tests

Executable tests can use hard-coded input, procedural input, or randomized input. Some hard-coded input is encoded directly into tests ource code, and some hard-coded input is serialized into JSON files.

Some of the tests with procedurally generated data have the ability to serialize failures to JSON. The pipeline detects these failures and saves the input that led to the failure case to AWS. These failure cases can be pulled from AWS, fixed, and added as hard-coded input for future pipeline executions.

## Visual tests

Visual tests usually directly execute some algorithm against some UI-configurable input. As an example, the GJK and EPA algorithm visual tests let the user select shapes and transforms to run the two algorithms on. This relies on the user exploiting their intuition and understanding of the algorithm under test to detect bad configurations.

Some of the visual tests allow the user to serialize inputs that lead to incorrect output. These serialized inputs (JSON) can be used as hard-coded input for executable tests.

# Running viz and exec tests

The repo must be compiled with unit tests enabled. Add `-DBUILD_TESTS=1` to your `cmake` configuration string to build all of the available tests.

| Test type | Binary folder | Run from |
| --------- | ------------- | -------- |
| `exec_tests` | `build/exec_tests` | `build` directory |
| `viz_tests` | `build/bin` | `build` directory |

`viz_tests` require OpenGL. All of the visual test binaries put into the `bin` directory in the build folder. `exec_test` binaries are put into the `exec_tests` directory in the build folder. Be sure to run the any tests (visual or executable) directly from the build folder (e.g. don't `cd bin` or `cd exec_tests`).