#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <string>
using std::vector;
using namespace std::string_literals;

namespace mnist_reader
{
	inline std::istream& operator %(std::istream& s, int32_t& v)
	{
		s.read(reinterpret_cast<char*>(&v), sizeof(v));
		std::reverse(reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v + 1));
		return s;
	}

	static std::tuple<vector<vector<uint8_t>>, vector<uint8_t>> read(std::string_view image_file,
	                                                                 std::string_view label_file)
	{
		std::ifstream ifs_images(image_file.data(), std::ifstream::in | std::ifstream::binary);
		if (!ifs_images.is_open()) throw std::runtime_error("Can't open image file '"s + image_file.data() + "'."s);

		int32_t magic;
		int32_t num_images;
		int32_t num_rows;
		int32_t num_cols;

		ifs_images % magic % num_images % num_rows % num_cols;

		if (magic != 2051) throw std::runtime_error("'"s + label_file.data() + "' - wrong file format"s);

		std::ifstream ifs_labels(label_file.data(), std::ifstream::in | std::ifstream::binary);
		if (!ifs_labels.is_open()) throw std::runtime_error("Can't open image file '"s + label_file.data() + "'."s);

		int32_t num_labels;

		ifs_labels % magic % num_labels;

		if (magic != 2049) throw std::runtime_error("'"s + label_file.data() + "' - wrong file format"s);

		vector<vector<uint8_t>> images(num_images, vector<uint8_t>(num_rows * num_cols));
		vector<uint8_t> labels(num_labels);

		for (auto i = 0; i < num_images; ++i)
		{
			ifs_images.read(reinterpret_cast<char*>(&images[i][0]), num_rows * num_cols);
			ifs_labels.read(reinterpret_cast<char*>(&labels[i]), 1);
		}

		return {images, labels};
	}

    static vector<uint8_t> readLable(std::string_view label_file)
    {


        std::ifstream ifs_labels(label_file.data(), std::ifstream::in | std::ifstream::binary);
        if (!ifs_labels.is_open()) throw std::runtime_error("Can't open image file '"s + label_file.data() + "'."s);

        int32_t num_labels;
        int32_t magic;
        ifs_labels % magic % num_labels;

        vector<uint8_t> labels(num_labels);


        for (auto i = 0; i < num_labels; ++i)
        {
            ifs_labels.read(reinterpret_cast<char*>(&labels[i]), 1);
        }

        return  labels;
    }

    static vector<vector<uint8_t>> readImage(std::string_view image_file)
    {
        std::ifstream ifs_images(image_file.data(), std::ifstream::in | std::ifstream::binary);
        if (!ifs_images.is_open()) throw std::runtime_error("Can't open image file '"s + image_file.data() + "'."s);

        int32_t magic;
        int32_t num_images;
        int32_t num_rows;
        int32_t num_cols;

        ifs_images % magic % num_images % num_rows % num_cols;


        vector<vector<uint8_t>> images(num_images, vector<uint8_t>(num_rows * num_cols));

        for (auto i = 0; i < num_images; ++i)
        {
            ifs_images.read(reinterpret_cast<char*>(&images[i][0]), num_rows * num_cols);
        }

        return images;
    }
};
