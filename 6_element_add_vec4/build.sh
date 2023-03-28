rm mandelbrot.png
rm -rf build
mkdir build
cd build
cmake ..
make -j4
cd ..

echo ""

./build/vulkan_app