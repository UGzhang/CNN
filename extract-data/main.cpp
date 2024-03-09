#include "utils.h"
#include "mnist_reader.h"
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]){

    if(argc != 4) exit(0);

    const std::string& inputPath =  argv[1];
    const std::string& outputPath = argv[2];
    const int index = atoi(argv[3]);
    std::ofstream output(outputPath);

    if(inputPath.find("idx3") != std::string::npos){
        auto imgs = mnist_reader::readImage(inputPath);
        auto img = utils::normalize_image_set(imgs)[index];
        output << 2 << "\n";
        output << 28 << "\n";
        output << 28 << "\n";

        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                output << img[i*28+j] << "\n";
            }
        }
        output << std::endl;

    } else{
        auto labels = mnist_reader::readLable(inputPath);
        auto label = utils::to_categorical(labels)[index];
        output << 1 << "\n";
        output << 10 << "\n";

        for (int i = 0; i < 10; ++i) {
            output << label[i] << "\n";
        }
        output << std::endl;
    }




}