package com.stolsvik.neurons

import groovy.transform.CompileStatic

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-13 23:34
 */
@CompileStatic
class Neuron_Sigmoid extends Neuron_WithInputs {

    Neuron_Sigmoid(Neuron[] incomingNeurons) {
        super(incomingNeurons)
    }

    /**
     * Implements the <i>standard logistic function<i>, as described
     * <a href="https://en.wikipedia.org/wiki/Logistic_function#Mathematical_properties">in Wikipedia</a>.
     */
    @Override
    double calculateOutputValue() {
        double t = calculateRawOutputValue()
        // Sigmoid, specifically 'standard logistic function'.
        1 / (1 + Math.exp(-t))
    }

    /**
     * NOTICE: Assumes that the neuron is already {@link Neuron#calculate()}'ed, as it uses the
     * {@link #getOutputValue() output value}.
     *
     * Implements the derivative of the <i>standard logistic function<i>, as described
     * <a href="https://en.wikipedia.org/wiki/Logistic_function#Derivative">in Wikipedia</a>.
     */
    double getDerivativeOfOutputValue() {
        // Since dy/dt for StandardLogisticFunction (SLF) ends up being 'SLF(1-SLF)', we use the previously calculated output value.
        outputValue * (1 - outputValue)
    }
}
