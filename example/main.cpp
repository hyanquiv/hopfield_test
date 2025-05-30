#include "hopfield.h"
#include "mnist/mnist_reader.hpp"
#include <unordered_map>
#include <set>

std::vector<int> binarize_image(const std::vector<uint8_t>& img) {
    std::vector<int> binary(img.size());
    for (size_t i = 0; i < img.size(); ++i) {
        binary[i] = img[i] > 127 ? 1 : -1;
    }
    return binary;
}

int main() {
    const size_t img_size = 28 * 28;

    // Cargar Fashion-MNIST desde el directorio
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data");
    std::cout << "Train size: " << dataset.training_images.size() << std::endl;

    HopfieldNetwork network(img_size);

    // Seleccionar 1 imagen por clase (0-9)
    std::unordered_map<uint8_t, std::vector<int>> patterns;
    std::set<uint8_t> classes_found;

    for (size_t i = 0; i < dataset.training_images.size(); ++i) {
        uint8_t label = dataset.training_labels[i];
        if (classes_found.find(label) == classes_found.end()) {
            patterns[label] = binarize_image(dataset.training_images[i]);
            classes_found.insert(label);
            if (classes_found.size() == 10) break;
        }
    }

    // Entrenar con 10 patrones
    std::vector<std::vector<int>> pattern_list;
    for (auto it = patterns.begin(); it != patterns.end(); ++it) {
        const uint8_t& label = it->first;
        const std::vector<int>& pattern = it->second;
        pattern_list.push_back(pattern);
    }
    network.train(pattern_list);

    // Probar recuperación
    auto test_img = binarize_image(dataset.test_images[0]);
    auto recovered = network.recall(test_img);

    // Mostrar diferencia entre imagen original y recuperada (opcional)
    int diff = 0;
    for (size_t i = 0; i < img_size; ++i) {
        if (test_img[i] != recovered[i]) ++diff;
    }
    std::cout << "Diferencias tras recuperación: " << diff << " de " << img_size << std::endl;

    return 0;
}
