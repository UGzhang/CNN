echo "This script should build your project now..."
#!/bin/bash
mkdir build
cd build
cmake ..
make
rm -rf build