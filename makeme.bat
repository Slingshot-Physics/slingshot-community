@echo off

if "%1" == "--help" goto:SHOW_HELP

pushd .
mkdir build
cd build
cmake -G "MinGW Makefiles" %* ..
mingw32-make -j4
popd
goto:eof

:SHOW_HELP
cmake -DHELP_SOS=1
