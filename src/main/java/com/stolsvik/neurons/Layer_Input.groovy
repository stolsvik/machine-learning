package com.stolsvik.neurons

import groovy.transform.CompileStatic;

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-15 21:32
 */
@CompileStatic
class Layer_Input implements Layer {

    Neuron_Input[] neurons

    Neuron[] getNeurons() {
        return neurons
    }

    private Layer_Input(int size) {
        neurons = new Neuron_Input[size]
        for (int i = 0; i<size; i++) {
            neurons[i] = new Neuron_Input()
        }
    }

    static Layer_Input createInputLayer(int size) {
        new Layer_Input(size)
    }

    void setInputs(double[] values) {
        for (int i = 0; i< values.length; i++) {
            neurons[i].outputValue = values[i] / 255d
        }
    }

    @Override
    void calculate() {
        /* no-op */
    }
}
