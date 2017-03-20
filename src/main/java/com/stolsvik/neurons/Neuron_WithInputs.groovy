package com.stolsvik.neurons

import groovy.transform.CompileStatic
import groovy.util.logging.Slf4j

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-13 23:27
 */
@CompileStatic
@Slf4j
abstract class Neuron_WithInputs implements Neuron {
    double[] weights
    double bias
    Neuron[] incoming

    // Cached calculated output value
    double outputValue

    Neuron_WithInputs(Neuron[] incoming) {
        this.incoming = incoming
        weights = new double[this.incoming.length]
    }

    @Override
    void calculate() {
        outputValue = calculateOutputValue()
    }

    double δ_nodeDelta

    void updateWithNodeDelta(double δ_nodeDelta) {
        double eta = 0.00001d
        this.δ_nodeDelta = δ_nodeDelta
        for (int i = 0; i < weights.length; i++) {
            weights[i] = weights[i] + (eta * incoming[i].outputValue * δ_nodeDelta)
        }
        bias = bias + (eta * 1d * δ_nodeDelta)
    }

    /**
     * Must be implemented by extensions. Feel free to use the method {@link #calculateRawOutputValue()}.
     * @return the output value of this neuron - which is cached in the field {@link #outputValue} by {@link #calculate()}.
     */
    abstract double calculateOutputValue()

    double calculateRawOutputValue() {
        double Σ = 0
        for (int i = 0; i < incoming.length; i++) {
            Σ += (weights[i] * incoming[i].outputValue)
        }
        return Σ + bias
    }
}
