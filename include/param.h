#pragma once

struct Param{
    int num_epochs;
    int batch_size;
    int n_train;
    int dim_in;
    int hidden_size;
    int dim_out;
    double learning_rate;
    std::string rel_path_train_images;
    std::string rel_path_train_labels;
    std::string rel_path_test_images;
    std::string rel_path_test_labels;
    std::string rel_path_log_file;

};