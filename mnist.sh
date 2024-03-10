echo "This script should trigger the training and testing of your neural network implementation..."
#!/bin/bash
sh build.sh
exec ./AdvPT $1
