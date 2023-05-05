rmdir /s /q build
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
mingw32-make.exe -j4
cd ..
.\build\vulkan_app.exe