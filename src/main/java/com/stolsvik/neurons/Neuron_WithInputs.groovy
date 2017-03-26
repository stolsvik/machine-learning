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

    // Cached calculated output value for current input, forward pass
    double outputValue
    // Cached calculated node delta for current input, backpropagation pass.
    double δ_nodeDelta

    Neuron_WithInputs(Neuron[] incoming) {
        this.incoming = incoming
        weights = new double[this.incoming.length]
    }

    @Override
    void calculate() {
        outputValue = calculateOutputValue()
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

    /**
     * @return the derivative of the current output value.
     */
    abstract double getDerivativeOfOutputValue()

    double δ_nodeDelta_MiniBatchAccumulated

    void accumulateNodeDeltaForMinibatch() {
        δ_nodeDelta_MiniBatchAccumulated += δ_nodeDelta
    }

    void updateWeightsAndBiasWithMinibatchAccumulatedNodeDelta(double η_trainingRate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = weights[i] - (η_trainingRate * incoming[i].outputValue * δ_nodeDelta_MiniBatchAccumulated)
        }
        bias = bias - (η_trainingRate * 1d * δ_nodeDelta_MiniBatchAccumulated)
    }

    void clearAccumulatedNodeDelta() {
        δ_nodeDelta_MiniBatchAccumulated = 0
    }

    String dumpCurrentWeights() {
        StringBuilder buf = new StringBuilder()
        weights.eachWithIndex { double weight, int i ->
            Static.singleDumpElement(buf, i, Static.ff4(weight))
        }
        buf.toString().substring(0, buf.length() - 1)
    }
}
