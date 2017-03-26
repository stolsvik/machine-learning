package com.stolsvik.neurons

import groovy.transform.CompileStatic;

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-15 21:32
 */
@CompileStatic
class Layer_Input extends Layer_Abstract {

    Neuron_Input[] neurons_Input

    Neuron[] getNeurons() {
        return neurons_Input
    }

    private Layer_Input(int size) {
        super(0)
        neurons_Input = new Neuron_Input[size]
        for (int i = 0; i<size; i++) {
            neurons_Input[i] = new Neuron_Input()
        }
    }

    static Layer_Input createInputLayer(int size) {
        new Layer_Input(size)
    }

    void setInputs(double[] values) {
        for (int i = 0; i< values.length; i++) {
            neurons_Input[i].outputValue = values[i] / 255d - 0.5d
        }
    }

    @Override
    void calculate() {
        /* no-op */
    }
}
