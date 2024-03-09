echo "This script should trigger the training and testing of your neural network implementation..."
#!/bin/bash
mkdir build
cd build
cmake ..
make
cd ../
exec ./AdvPT $1
rm -rf build