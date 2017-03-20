package com.stolsvik.neurons

import groovy.transform.CompileStatic
import java.util.function.DoubleSupplier

/**
 * Resources:
 *
 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 *
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-14 00:19
 */
@CompileStatic
class NeuralNetwork {

    List<Layer> layers = []

    Layer_Input inputLayer
    Layer lastLayer
    Layer_WithInputs outputLayer

    void addLayer(Layer layer) {
        if (layers.size() == 0) {
            if (!layer instanceof Layer_Input) {
                throw new IllegalArgumentException("First layer must be ${Layer_Input.class.simpleName}.")
            }
            inputLayer = (Layer_Input) layer
        }
        else {
            if (!layer instanceof Layer_WithInputs) {
                throw new IllegalArgumentException("Other Layers than first must be ${Layer_WithInputs.class.simpleName} or subclass.")
            }
            outputLayer = (Layer_WithInputs) layer
        }
        lastLayer = layer
        layers.add(layer)
    }

    void calculate() {
        // Start from top, "Feed Forward"
        layers.each { it.calculate() }
    }

    void initialize(DoubleSupplier randomWeight, DoubleSupplier randomBias) {
        layers.eachWithIndex { layer, index ->
            if (index == 0) {
                return
            }
            Layer_WithInputs hiddenOrOutputLayer = (Layer_WithInputs) layer
            hiddenOrOutputLayer.initialize(randomWeight, randomBias)
        }
    }

}
