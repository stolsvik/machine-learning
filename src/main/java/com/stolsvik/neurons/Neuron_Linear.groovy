package com.stolsvik.neurons;

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-20 22:14
 */
class Neuron_Linear extends Neuron_WithInputs{

    Neuron_Linear(Neuron[] incomingNeurons) {
        super(incomingNeurons)
    }

    /**
     * Directly returns {@link #calculateRawOutputValue()}.
     */
    @Override
    double calculateOutputValue() {
        // f(x) = x
        calculateRawOutputValue()
    }

    @Override
    double getDerivativeOfOutputValue() {
        // f(x) = x -> f'(x) = 1
        return 1
    }
}
