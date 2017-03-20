package com.stolsvik.neurons

import groovy.transform.CompileStatic
import groovy.util.logging.Slf4j
import java.util.function.DoubleSupplier

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-14 00:08
 */
@CompileStatic
@Slf4j
class Layer_WithInputs implements Layer {

    Neuron_WithInputs[] neurons

    private double _size

    private Layer_WithInputs(Neuron_WithInputs[] neurons) {
        this.neurons = neurons
        _size = neurons.size()
    }

    static Layer_WithInputs createSigmoidLayer(int size, Neuron[] inputs) {
        Neuron_WithInputs[] neurons = new Neuron_WithInputs[size]
        for (int i = 0; i < size; i++) {
            neurons[i] = new Neuron_Sigmoid(inputs)
        }
        new Layer_WithInputs(neurons)
    }

    static Layer_WithInputs createLinearLayer(int size, Neuron[] inputs) {
        Neuron_WithInputs[] neurons = new Neuron_WithInputs[size]
        for (int i = 0; i < size; i++) {
            neurons[i] = new Neuron_Linear(inputs)
        }
        new Layer_WithInputs(neurons)
    }

    void initialize(DoubleSupplier randomWeight, DoubleSupplier randomBias) {
        for (int i = 0; i < neurons.length; i++) {
            Neuron_WithInputs neuron = neurons[i]
            for (int j = 0; j < neuron.incoming.length; j++) {
                neuron.weights[j] = randomWeight.getAsDouble()
                neuron.bias = randomBias.getAsDouble()
            }
        }
    }

    void calculate() {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].calculate()
        }
    }

    /**
     * Works directly on the {@link Neuron#getOutputValue()}, and hence doesn't rely on the softmax.
     * @return the index of the neuron with the highest output value.
     */
    int getIdxOfNeuronMaxOutputValue() {
        int idx = 0
        double max = neurons[0].outputValue
        for (int i = 1; i < neurons.size(); i++) {
            double outputValue = neurons[i].outputValue
            if (outputValue > max) {
                idx = i
                max = outputValue
            }
        }
        return idx
    }

    /**
     * Calculates the Softmax of the non-activation-function neurons, and stores them into the array parameter
     * 'softmaxTransfer'.
     *
     * <b>Note that the neurons in this layer should NOT have any non-linearities, e.g. Sigmoid/Logistic, ReLU etc.
     * applied as activation/transfer function, but rather have the plain summation values, i.e. (Σ(aj*wij))+bi, as
     * their output.</b>
     *
     * <i>Softmax is a generalization of the logistic function that "squashes" a K-dimensional vector 'z' of arbitrary
     * real values to a K-dimensional vector σ(z) of real values in the range (0, 1) that add up to 1.</i>
     *
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">Softmax function @ Wikipedia</a>
     */
    void calculateSoftmaxOutput(double[] softmaxTransfer) {
        int size = neurons.size()
        double Σek = 0
        for (int i = 0; i < size; i++) {
            double z = neurons[i].outputValue
            double ez = Math.exp(z)
            softmaxTransfer[i] = ez
            Σek += ez
        }
        for (int i = 0; i < size; i++) {
            softmaxTransfer[i] = softmaxTransfer[i] / Σek
        }
    }

    void softmaxAndBackpropagateErrors(double[] softmaxTransfer, int label) {
        calculateSoftmaxOutput(softmaxTransfer)
        for (int i = 0; i < neurons.length; i++) {
            double ai = softmaxTransfer[i]
            boolean correct = label == i
            // crossEntropy += crossEntropyOfSingle(ai, correct)
            if (correct) {
                // ti == 1
                neurons[i].updateWithNodeDelta(1 - ai)
            }
            else {
                // ti == 0
                neurons[i].updateWithNodeDelta(0 - ai)
            }
        }
    }

    /**
     * Implements the <i>cross entropy</i> between the output vs. the expected (labels), as described
     * <a href="https://en.wikipedia.org/wiki/Cross_entropy">in Wikipedia</a>.
     */
    double softmaxAndSumCrossEntropy(double[] softmaxTransfer, int label) {
        calculateSoftmaxOutput(softmaxTransfer)
        double crossEntropy = 0
        for (int i = 0; i < neurons.length; i++) {
            double ai = softmaxTransfer[i]
            crossEntropy += crossEntropyOfSingle(ai, i == label)
        }
        return crossEntropy
    }


    static double crossEntropyOfSingle(double a, boolean correct) {
        if (correct) {
            if (a == 0) {
                return 10 // e^-10 = 0,0000454
            }
            return -Math.log(a)
        }
        else {
            if (a == 1) {
                return 10 // The term in question is (1-t)*log(1-a) .. hence, if t=0, a=0, it will crash.
            }
            double ret = Math.log(1d - a)
            return ret == 0d ? -0d : -ret
        }
    }

    /**
     * NOTE: FUCKED
     *
     * NOTICE: This won't do softmax again, assuming that {@link #calculateSoftmaxOutput(double [ ])} - or equivalently,
     * {@link #getSparseCrossEntropy(double [], int)} - has been invoked previously.
     *
     * Implements the derivative of <i>cross entropy</i> between the output vs. the expected (labels), as described
     * <a href="http://www.themathpage.com/acalc/exponential.htm">in Wikipedia</a>.
     * also http://ltcconline.net/greenl/courses/116/ExpLog/logDerivative.htm
     * https://web.stanford.edu/group/pdplab/pdphandbook/handbookch6.html
     * https://www.ics.uci.edu/~pjsadows/notes.pdf
     * https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/
     */
    double getDerivativeOfCrossEntropy(double[] softmaxTransfer, int label) {
        // d/dx ln(x) => 1/x
        double predictedProbabilityForLabel = softmaxTransfer[label]
        1 / predictedProbabilityForLabel
    }

    @Override
    String toString() {
        StringBuilder buf = new StringBuilder()
        buf.append "${this.class.simpleName}.neurons.outputValue:\n"
        neurons.eachWithIndex { Neuron_WithInputs neuron, int i ->
            if (i > 0) {
                buf.append('\n')
            }
            buf.append "  Neuron #$i: ${Static.ff neuron.outputValue}, rawValue: ${Static.ff neuron.calculateRawOutputValue()}"
        }
        return buf.toString()
    }

    String currentOutputValues() {
        StringBuffer buf = new StringBuffer()
        neurons.eachWithIndex { Neuron_WithInputs neuron, int i ->
            buf.append(Static.ff(neuron.outputValue)).append(' ')
            if (i % 20 == 0) {
                buf.append('\n')
            }
        }
        buf.toString()
    }
}
