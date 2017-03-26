package com.stolsvik.neurons

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-26 11:41
 */
class Neuron_LeakyReLU extends Neuron_WithInputs {

    double negativeSlope

    Neuron_LeakyReLU(Neuron[] incomingNeurons, double negativeSlope) {
        super(incomingNeurons)
        this.negativeSlope = negativeSlope
    }

    /**
     * @return {@link #calculateRawOutputValue()} for x >= 0, and {@link #negativeSlope} x {@link #calculateRawOutputValue()} for x < 0
     */
    @Override
    double calculateOutputValue() {
        double x = calculateRawOutputValue()

        // f(x) = { x for x >= 0, x*negativeSlope for x < 0 }
        x >= 0 ? x : x * negativeSlope
    }

    /**
     * NOTICE: Assumes that the neuron is already {@link Neuron#calculate()}'ed, as it uses the
     * {@link #getOutputValue() output value}.
     *
     * @return constant 1 for x >= 0, and {@link #negativeSlope} for x < 0.
     */
    @Override
    double getDerivativeOfOutputValue() {
        // f'(x) = { 1 for x >= 0, negativeSlope for x < 0 }
        // Notice that I use the outputValue [i.e. f(x)] instead of the actual rawOutputValue [i.e. x], since I only
        // care about the sign, and x and f(x) has the same sign all the time.
        outputValue >= 0 ? 1 : negativeSlope
    }

}
