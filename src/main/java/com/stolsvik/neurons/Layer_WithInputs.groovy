package com.stolsvik.neurons

import groovy.transform.CompileStatic
import groovy.util.logging.Slf4j
import java.util.function.DoubleSupplier

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-14 00:08
 */
@CompileStatic
@Slf4j
class Layer_WithInputs extends Layer_Abstract {

    Neuron_WithInputs[] neurons_WithInput

    Neuron[] getNeurons() {
        return neurons_WithInput
    }

    private Layer_WithInputs(int layerIdx, Neuron_WithInputs[] neurons_WithInput) {
        super(layerIdx)
        this.neurons_WithInput = neurons_WithInput
    }

    static Layer_WithInputs createSigmoidLayer(int layerIdx, int size, Neuron[] inputs) {
        Neuron_WithInputs[] neurons = new Neuron_WithInputs[size]
        for (int i = 0; i < size; i++) {
            neurons[i] = new Neuron_Sigmoid(inputs)
        }
        new Layer_WithInputs(layerIdx, neurons)
    }

    static Layer_WithInputs createLinearLayer(int layerIdx, int size, Neuron[] inputs) {
        Neuron_WithInputs[] neurons = new Neuron_WithInputs[size]
        for (int i = 0; i < size; i++) {
            neurons[i] = new Neuron_Linear(inputs)
        }
        new Layer_WithInputs(layerIdx, neurons)
    }

    static Layer_WithInputs createLeakyReLULayer(int layerIdx, int size, double negativeSlope, Neuron[] inputs) {
        Neuron_WithInputs[] neurons = new Neuron_WithInputs[size]
        for (int i = 0; i < size; i++) {
            neurons[i] = new Neuron_LeakyReLU(inputs, negativeSlope)
        }
        new Layer_WithInputs(layerIdx, neurons)
    }

    void initialize(DoubleSupplier randomWeight, DoubleSupplier randomBias) {
        for (int i = 0; i < neurons_WithInput.length; i++) {
            Neuron_WithInputs neuron = neurons_WithInput[i]
            for (int j = 0; j < neuron.incoming.length; j++) {
                neuron.weights[j] = randomWeight.getAsDouble()
                neuron.bias = randomBias.getAsDouble()
            }
        }
    }

    void calculate() {
        for (int i = 0; i < neurons_WithInput.length; i++) {
            neurons_WithInput[i].calculate()
        }
    }

    /**
     * Works directly on the {@link Neuron#getOutputValue()}, and hence doesn't rely on the softmax.
     * @return the index of the neuron with the highest output value.
     */
    int getIdxOfNeuronMaxOutputValue() {
        int idx = 0
        double max = neurons_WithInput[0].outputValue
        for (int i = 1; i < neurons_WithInput.size(); i++) {
            double outputValue = neurons_WithInput[i].outputValue
            if (outputValue > max) {
                idx = i
                max = outputValue
            }
        }
        return idx
    }

    /**
     * Calculates the <i>Softmax</i> of the neurons in this layer (which should be of the {@link Neuron_Linear} type),
     * and stores them into the array parameter 'softmaxTransfer'.
     *
     * <b>Note that the neurons in this layer should NOT have any non-linearities, e.g. Sigmoid/Logistic, ReLU etc.
     * applied as activation/transfer function, but rather have the plain summation values, i.e. (Σ(aj*wij))+bi, as
     * their output, i.e. be of the type {@link Neuron_Linear}.</b>
     *
     * <i>Softmax is a generalization of the logistic function that "squashes" a K-dimensional vector 'z' of arbitrary
     * real values to a K-dimensional vector σ(z) of real values in the range (0, 1) that add up to 1.</i>
     *
     * @see <a href="https://en.wikipedia.org/wiki/Softmax_function">Softmax function @ Wikipedia</a>
     */
    void calculateSoftmaxOutput(double[] softmaxTransfer) {
        int size = neurons_WithInput.size()
        double Σek = 0
        for (int i = 0; i < size; i++) {
            double z = neurons_WithInput[i].outputValue
            double ez = Math.exp(z)
            softmaxTransfer[i] = ez
            Σek += ez
        }
        for (int i = 0; i < size; i++) {
            softmaxTransfer[i] = softmaxTransfer[i] / Σek
        }
    }

    /**
     * <b>For the Output layer only<b>: Calculate the <i>Node Delta</i> for the neurons in the this (output) layer,
     * by doing softmax on the neuron's (linear) output..
     *
     * @param softmaxTransfer the array in which to store the softmax.
     * @param target the "Target value" for the specimen being presented to the network now (i.e. for the image of a "7",
     * the value should be 7, i.e. the "label" of the image).
     */
    void softmaxAndCalculateCrossEntropyNodeDeltaForOutputLayerNeurons(double[] softmaxTransfer, int target) {
        calculateSoftmaxOutput(softmaxTransfer)
        for (int i = 0; i < neurons_WithInput.length; i++) {
            double ai = softmaxTransfer[i]

            /**
             * NodeDelta for CrossEntropy with Softmax on Output Layer: dCE/dai * dai/dnet = -(ti - ai)
             * If target is the digit 7, then ti = 0 for i != 7, and t7 = 1. (i.e. "One Hot" encoding of targets)
             */

            int ti = (target == i) ? 1 : 0

            neurons_WithInput[i].δ_nodeDelta = -(ti - ai)
        }
    }

    /**
     * Sum up the weighted sum of the node deltas which each of the neurons in this layer projects to (i.e. the neurons
     * which have these neuron as one of their inputs), then multiplies that with the derivative of this node's activation.
     *
     * @param nextLayer
     */
    void backpropagateNodeDeltasFromNextLayerIntoThisLayer(Layer_WithInputs nextLayer) {
        int thisLayerNeuronLength = neurons_WithInput.length
        int nextLayerNeuronLength = nextLayer.neurons_WithInput.length
        double[] weightedSumOfNodeDeltasFromNextLayer = new double[thisLayerNeuronLength]

        /**
         * DO REMEMBER: For this backprop summing, it is not THIS Layer l's weights W we're using now, it is weights of
         * Layer l+1. Hence, in the Wij notation, we're "From" node j, while the projected-to node is the "To" node i,
         * and the Wij notation refers to the weight FROM this Layer l's node j, TO Layer l+1's node i.
         * Again: the weight belongs to Layer l+1, not this Layer l.
         */

        for (int i = 0; i < nextLayerNeuronLength; i++) {
            Neuron_WithInputs projectedToNeuron = nextLayer.neurons_WithInput[i]
            double δ_nodeDelta_ForProjectedToNeuron = projectedToNeuron.δ_nodeDelta
            for (int j = 0; j < thisLayerNeuronLength; j++) {
                // OPTIMIZE: Cache the multiplied outputValue * weights in the projectedToNeuron from the forward pass?
                weightedSumOfNodeDeltasFromNextLayer[j] += δ_nodeDelta_ForProjectedToNeuron * projectedToNeuron.weights[j]
            }
        }

        for (int j = 0; j < thisLayerNeuronLength; j++) {
            Neuron_WithInputs thisNeuron = this.neurons_WithInput[j]
            thisNeuron.δ_nodeDelta = weightedSumOfNodeDeltasFromNextLayer[j] * thisNeuron.derivativeOfOutputValue
        }
    }

    void accumulateNodeDeltaForMiniBatch() {
        for (int j = 0; j < neurons_WithInput.length; j++) {
            this.neurons_WithInput[j].accumulateNodeDeltaForMinibatch()
        }
    }

    void updateWeightsAndBiasWithMinibatchAccumulatedNodeDelta(double η_trainingRate) {
        for (int i = 0; i < neurons_WithInput.length; i++) {
            neurons_WithInput[i].updateWeightsAndBiasWithMinibatchAccumulatedNodeDelta(η_trainingRate)
        }
    }

    void clearAccumulatedNodeDelta() {
        for (int i = 0; i < neurons_WithInput.length; i++) {
            neurons_WithInput[i].clearAccumulatedNodeDelta()
        }
    }

    /**
     * Implements the <i>cross entropy</i> between the output vs. the expected (labels), as described
     * <a href="https://en.wikipedia.org/wiki/Cross_entropy">in Wikipedia</a>.
     */
    double softmaxAndSumCrossEntropyForOutputNeuronsVsTarget(double[] softmaxTransfer, int label) {
        calculateSoftmaxOutput(softmaxTransfer)
        double crossEntropy = 0
        for (int i = 0; i < neurons_WithInput.length; i++) {
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
     *
     * Explain Backprop:
     * https://ayearofai.com/rohan-lenny-1-neural-networks-the-backpropagation-algorithm-explained-abf4609d4f9d
     * http://adbrebs.github.io/Backpropagation-simply-explained/
     * http://neuralnetworksanddeeplearning.com/chap2.html
     * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
     * https://web.stanford.edu/group/pdplab/pdphandbook/handbookch6.html
     */
    double getDerivativeOfCrossEntropy(double[] softmaxTransfer, int label) {
        // d/dx ln(x) => 1/x
        double predictedProbabilityForLabel = softmaxTransfer[label]
        1 / predictedProbabilityForLabel
    }

    String dumpCurrentWeights(int neuronIdx) {
        StringBuilder buf = new StringBuilder()
        double[] weights = neurons_WithInput[neuronIdx].weights
        weights.eachWithIndex { double weight, int i ->
            Static.singleDumpElement(buf, i, Static.ff4(weight))
        }
        buf.toString().substring(0, buf.length() - 1)
    }

    String dumpCurrentNodeDeltas() {
        StringBuilder buf = new StringBuilder()
        neurons_WithInput.eachWithIndex { Neuron_WithInputs neuron, int i ->
            Static.singleDumpElement(buf, i, Static.ff4(neuron.δ_nodeDelta))
        }
        buf.toString().substring(0, buf.length() - 1)
    }

    String dumpCurrentAccumulatedNodeDeltas() {
        StringBuilder buf = new StringBuilder()
        neurons_WithInput.eachWithIndex { Neuron_WithInputs neuron, int i ->
            Static.singleDumpElement(buf, i, Static.ff4(neuron.δ_nodeDelta_MiniBatchAccumulated))
        }
        buf.toString().substring(0, buf.length() - 1)
    }
}
