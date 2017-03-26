package com.stolsvik.neurons

import com.stolsvik.neurons.mnist.MnistImages
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

    // ===== SETUP METHODS

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
            // Set up next-layer pointer for last layer (next is the one we're adding, last is the one before)
            ((Layer_Abstract) lastLayer).nextLayer = (Layer_WithInputs) layer
        }
        lastLayer = layer
        layers.add(layer)
    }

    List<Layer_WithInputs> hiddenLayers

    List<Layer_WithInputs> hiddenLayers_reversed

    List<Layer_WithInputs> hiddenAndOutputLayers

    double[] inputTransfer
    double[] softmaxTransfer

    void initialize(DoubleSupplier randomWeight, DoubleSupplier randomBias) {
        // :: Pick out interesting subsets of Layers
        // ..Hidden and Output: Exclude Input. Input, Hidden, Output -> Size=3 -> Range=[1,2], i.e. [1, size()-1]
        hiddenAndOutputLayers = layers[1..layers.size() - 1] as List<Layer_WithInputs>

        // ..Hidden Layers: Exclude Input and Output. Input, Hidden, Output -> Size=3 -> Range=[1,1], i.e. [1, size()-2]
        hiddenLayers = layers[1..layers.size() - 2] as List<Layer_WithInputs>

        // ..Reverse of Hidden (used for backprop)
        hiddenLayers_reversed = hiddenLayers.reverse()

        // :: Create the "transfer arrays", which is a way to avoid creating objects while training & evaluating
        inputTransfer = new double[inputLayer.neurons.length]
        softmaxTransfer = new double[outputLayer.neurons.length]

        // :: Initialize Weights and Biases with provided Random number generators.
        hiddenAndOutputLayers.each { it.initialize(randomWeight, randomBias) }
    }

    // ===== TRAINING METHODS

    void train(MnistImages trainingImages, MnistImages testImages, int miniBatchSize, double η_trainingRate) {
        println '\n========= TRAINING ==========\n'

        for (int epoch = 0; epoch < 10000; epoch++) {
            trainEpoch(epoch, trainingImages, testImages, miniBatchSize, η_trainingRate)
        }
        assessCrossEntropyCostAndAccuracy(this, trainingImages)
    }

    private
    void trainEpoch(int epoch, MnistImages trainingImages, MnistImages testImages, int miniBatchSize, double η_trainingRate) {
        int batches = (int) (trainingImages.size / miniBatchSize)

        println '------------------------------------------------------------------------------------------------------'
        println "TRAINING Epoch $epoch, mini-batch size $miniBatchSize," +
                " training rate ${sprintf('%f', η_trainingRate)}, for $this"
        long startEpochNanos = System.nanoTime()

        // :: Randomize the training images before each epoch
        trainingImages.randomize()

        // :: Do the actual bunch of mini batches
        for (int i = 0; i < batches; i++) {
            trainMiniBatch(trainingImages, i * miniBatchSize, miniBatchSize, η_trainingRate)
        }

        // :: Epoch finished, some some stats
        println "  \\- Epoch $epoch done, took ${Static.timingFrom(startEpochNanos)}."
        infoAboutSingleImage(this, trainingImages, 0)
        dumpNetwork(this)
        assessCrossEntropyCostAndAccuracy(this, testImages)
        if (epoch % 10 == 0) {
            println '### TRAINING SET'
            assessCrossEntropyCostAndAccuracy(this, trainingImages)
        }
        println ''
    }

    private
    void trainMiniBatch(MnistImages images, int from, int miniBatchSize, double η_trainingRate) {
        // :: Clear the accumulated node deltas, to prepare for new accumulation for minibatch
        for (int l = 1; l < this.layers.size(); l++) {
            Layer_WithInputs layer = (Layer_WithInputs) this.layers[l]
            layer.clearAccumulatedNodeDelta()
        }

        // :: Feed forward input values through the network, then accumulate node deltas in backpropagation
        for (int i = from; i < from + miniBatchSize; i++) {

            // :: FEED FORWARD THE INPUT VALUES THROUGH THE NETWORK

            // Get input values (one image) and set them on the input neurons
            images.getImage(i, inputTransfer)
            this.inputLayer.setInputs(inputTransfer)
            // Calculate all layers, from input to output
            this.hiddenAndOutputLayers.each { l -> l.calculate() }

            // :: CALCULATE ERROR / NODE DELTA FOR OUTPUT NODES

            // Calculate the "error": i.e. find the node delta for the output layer. (We do not actually calculate the error function, only its derivative).
            this.outputLayer.softmaxAndCalculateCrossEntropyNodeDeltaForOutputLayerNeurons(softmaxTransfer, images.getLabel(i))

            // :: BACK PROPAGATE THE ERRORS BACKWARDS THROUGH THE NETWORK

            // Do backpropagation of node deltas for this sample (Go backwards through layers: Layer L-1 takes values from L, Layer L-2 takes values from Layer L-1...)
            this.hiddenLayers_reversed.each { l -> l.backpropagateNodeDeltasFromNextLayerIntoThisLayer(l.nextLayer) }

            // .. then accumulate the node deltas into the accumulator (simple layer-by-layer, only involving values from the layer itself)
            this.hiddenAndOutputLayers.each { l -> l.accumulateNodeDeltaForMiniBatch() }
        }
        // :: Apply the accumulated node deltas to the weights
        this.hiddenAndOutputLayers.each { l -> l.updateWeightsAndBiasWithMinibatchAccumulatedNodeDelta(η_trainingRate) }
    }

    // ===== INTROSPECTION METHODS

    static void assessCrossEntropyCostAndAccuracy(NeuralNetwork network, MnistImages images) {
        println "Evaluating [$images.size $images.type]:"
        long nanosStart = System.nanoTime()
        double summedCrossEntropy = 0
        double[] softmaxTransfer = new double[network.outputLayer.neurons.length]
        int correct = 0
        double[] inputTransfer = new double[28 * 28]
        for (int i = 0; i < images.size; i++) {
            images.getImage(i, inputTransfer)
            network.inputLayer.setInputs(inputTransfer)
            network.hiddenAndOutputLayers.each { l -> l.calculate() }

            int label = images.getLabel(i)
            summedCrossEntropy += network.outputLayer.softmaxAndSumCrossEntropyForOutputNeuronsVsTarget(softmaxTransfer, label)

            int predictedLabel = network.outputLayer.getIdxOfNeuronMaxOutputValue()
            if (predictedLabel == label) {
                correct++
            }
        }

        println " Done, time: ${Static.timingFrom(nanosStart)}"
        println " Σsamples(crossEntropy): ${Static.ff(summedCrossEntropy)}"
        println " ACCURACY: $correct out of $images.size, which is ${(correct / (double) images.size) * 100}%"
    }

    static void infoAboutSingleImage(NeuralNetwork network, MnistImages trainingImages, int imageNumber) {
        int label = trainingImages.getLabel(imageNumber)

        // Calculate for image imageNumber
        double[] inputTransfer = new double[28 * 28]
        trainingImages.getImage(imageNumber, inputTransfer)
        network.inputLayer.setInputs(inputTransfer)
        network.hiddenAndOutputLayers.each { l -> l.calculate() }

        int outputLayerSize = network.outputLayer.neurons.length

        int predictedLabel = network.outputLayer.getIdxOfNeuronMaxOutputValue()

        println "Info about image #$imageNumber, which has label $label, predicted to $predictedLabel," +
                " which is ${label == predictedLabel ? '!!CORRECT!!' : 'incorrect'}."

        double[] oneHotLabelTransfer = new double[outputLayerSize]
        trainingImages.getOneHotLabel(imageNumber, oneHotLabelTransfer)

        double[] softmaxTransfer = new double[outputLayerSize]

        // Calculating node deltas for the output layer for this image.
        // NOTICE: THIS WILL NOT TRAIN THE NETWORK SINCE WE DO *NOT* UPDATE THE WEIGHTS!!!
        network.outputLayer.softmaxAndCalculateCrossEntropyNodeDeltaForOutputLayerNeurons(softmaxTransfer, label)
        double sum = 0
        softmaxTransfer.eachWithIndex { double value, int i ->
            Neuron_WithInputs neuron = (Neuron_WithInputs) network.outputLayer.neurons[i]
            double ln = Layer_WithInputs.crossEntropyOfSingle(value, label == i)
            sum += ln
            println "  Neuron #$i: t: ${(int) oneHotLabelTransfer[i]}, a: ${Static.ff(neuron.outputValue)}," +
                    " softmax(a): ${Static.ff(value)}, crossEntropy: ${Static.ff(ln)}," +
                    " δ_nodeDelta:${Static.ff(neuron.δ_nodeDelta)} # rawValue:${neuron.calculateRawOutputValue()}"
        }
        println "  Σneurons(crossEntropy): ${sum}"
        double sparseCrossEntropy = network.outputLayer.softmaxAndSumCrossEntropyForOutputNeuronsVsTarget(softmaxTransfer, label)
        println "  vs. sparse cross entropy: $sparseCrossEntropy"
    }

    static void dumpNetwork(NeuralNetwork network) {
        network.dumpWeights(0)
        println ''
        network.dumpNodeDeltas()
        println ''
        network.dumpAccumulatedNodeDeltas()
        println ''
        network.dumpOutputValues()
        println ''
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

    @Override
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
}
