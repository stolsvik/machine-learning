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


    String toString() {
        StringBuilder buf = new StringBuilder()
        buf.append("Neural Network of ${layers.size()} layers: ")
        layers.eachWithIndex { Layer layer, int i ->
            if (i > 0) {
                buf.append(', ')
            }
            buf.append("#$i:[${layer.neurons.size()}]*${layer.neurons[0].getClass().getSimpleName()}")
        }
        buf.toString()
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

    void dumpOutputValues() {
        for (int l = 0; l < layers.size(); l++) {
            Layer_Abstract layer = (Layer_Abstract) layers[l]
            println "Layer $l output values:\n${layer.dumpCurrentOutputValues()}"
        }
    }

    void dumpNodeDeltas() {
        for (int l = 1; l < layers.size(); l++) {
            Layer_WithInputs layer = (Layer_WithInputs) layers[l]
            println "Layer $l node deltas:\n${layer.dumpCurrentNodeDeltas()}"
        }
    }

    void dumpAccumulatedNodeDeltas() {
        for (int l = 1; l < layers.size(); l++) {
            Layer_WithInputs layer = (Layer_WithInputs) layers[l]
            println "Layer $l ACCUMULATED node deltas (from last mini batch):\n${layer.dumpCurrentAccumulatedNodeDeltas()}"
        }
    }

    void dumpWeights(int neuronIdxOfEachLayer) {
        for (int l = 1; l < layers.size(); l++) {
            Layer_WithInputs layer = (Layer_WithInputs) layers[l]
            println "Layer $l weights of neuron $neuronIdxOfEachLayer:\n" +
                    "${layer.dumpCurrentWeights(neuronIdxOfEachLayer)}"
        }
    }

}
