#include <iostream>
#include <unordered_map>
#include <set>
#include <vector>
#include "mnist/mnist_reader_less.hpp"
#include "hopfield.h"

std::vector<int> binarize(const std::vector<uint8_t>& image) {
    std::vector<int> binary(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        binary[i] = image[i] > 127 ? 1 : -1;
    }
    return binary;
}

void print_image(const std::vector<int>& img) {
    for (size_t i = 0; i < img.size(); ++i) {
        std::cout << (img[i] > 0 ? "#" : ".");
        if ((i + 1) % 28 == 0)
            std::cout << '\n';
    }
}

int main() {
    constexpr size_t image_size = 28 * 28;

    // Cargar el dataset usando el reader "less"
    auto training_images = mnist::read_training_images<>();
    auto training_labels = mnist::read_training_labels<>();
    auto test_images = mnist::read_test_images<>();
    auto test_labels = mnist::read_test_labels<>();

    std::cout << "Datos cargados: " << training_images.size() << " imágenes de entrenamiento.\n";

    // Construir red Hopfield
    HopfieldNetwork net(image_size);

    // Tomar un patrón por clase
    std::unordered_map<uint8_t, std::vector<int>> patrones;
    std::set<uint8_t> clases_encontradas;

    for (size_t i = 0; i < training_images.size(); ++i) {
        uint8_t etiqueta = training_labels[i];
        if (clases_encontradas.count(etiqueta) == 0) {
            patrones[etiqueta] = binarize(training_images[i]);
            clases_encontradas.insert(etiqueta);
            if (clases_encontradas.size() == 10) break;
        }
    }

    std::vector<std::vector<int>> lista_patrones;
    for (auto& [label, pattern] : patrones) {
        lista_patrones.push_back(pattern);
    }

    net.train(lista_patrones);
    std::cout << "Red entrenada con 10 patrones.\n";

    // Usar una imagen de test como entrada
    size_t test_idx = 0;
    auto imagen_original = binarize(test_images[test_idx]);
    auto imagen_recuperada = net.recall(imagen_original);

    std::cout << "\nImagen original:\n";
    print_image(imagen_original);

    std::cout << "\nImagen recuperada:\n";
    print_image(imagen_recuperada);

    return 0;
}


