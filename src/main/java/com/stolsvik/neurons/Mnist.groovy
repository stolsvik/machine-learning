package com.stolsvik.neurons

import com.stolsvik.neurons.mnist.MnistImages
import groovy.transform.CompileStatic
import java.util.function.DoubleSupplier

/**
 * Backprop: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example
 *
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-14 00:16
 */
@CompileStatic
class Mnist {

    static void main(String... args) {
        NeuralNetwork network = new NeuralNetwork()
        network.addLayer(Layer_Input.createInputLayer(28 * 28))
//        network.addLayer(Layer_WithInputs.createSigmoidLayer(500, network.lastLayer.neurons))
        network.addLayer(Layer_WithInputs.createSigmoidLayer(network.layers.size(), 250, network.lastLayer.neurons))
        network.addLayer(Layer_WithInputs.createLinearLayer(network.layers.size(), 10, network.lastLayer.neurons))

        Random random = new Random()
        random.setSeed(2l)
        network.initialize({ return random.nextGaussian() * 0.1d } as DoubleSupplier,
                { return random.nextGaussian() * 0.1d } as DoubleSupplier)

        MnistImages trainingImages = MnistImages.training
        MnistImages testImages = MnistImages.test

        println "\n========= INITIAL VALUES, before any training ========="
        infoAboutSingleImage(network, trainingImages, 0)
        dumpNetwork(network)

        assessCrossEntropyCostAndAccuracy(network, testImages)
        // assessCrossEntropyCostAndAccuracy(network, trainingImages)

        train(network, trainingImages, testImages)
    }

    private static void train(NeuralNetwork network, MnistImages trainingImages, MnistImages testImages) {
        println '\n========= TRAINING ==========\n'
        double[] inputTransfer = new double[28 * 28]
        double[] softmaxTransfer = new double[network.outputLayer.neurons.size()]

        boolean doLog = false

        for (int epoch = 0; epoch < 10000; epoch++) {
            trainEpoch(epoch, network, trainingImages, inputTransfer, softmaxTransfer, doLog, testImages)
        }
        assessCrossEntropyCostAndAccuracy(network, trainingImages)
    }

    private
    static void trainEpoch(int epoch, NeuralNetwork network, MnistImages trainingImages, double[] inputTransfer, double[] softmaxTransfer, boolean doLog, MnistImages testImages) {
        int miniBatchSize = 100
        int batches = (int) (trainingImages.size / miniBatchSize)
        double η_trainingRate = 0.0001d

        println '------------------------------------------------------------------------------------------------------'
        println "TRAINING Epoch $epoch, mini-batch size $miniBatchSize," +
                " training rate ${sprintf('%f', η_trainingRate)}, for $network"
        long startEpochNanos = System.nanoTime()
        trainingImages.randomize()
        for (int i = 0; i < batches; i++) {
            trainMiniBatch(inputTransfer, softmaxTransfer, network, trainingImages, i * miniBatchSize, miniBatchSize, η_trainingRate, doLog)
        }
        println "  \\- Epoch $epoch done, took ${Static.timingFrom(startEpochNanos)}."
        infoAboutSingleImage(network, trainingImages, 0)
        dumpNetwork(network)
        assessCrossEntropyCostAndAccuracy(network, testImages)
        if (epoch % 10 == 0) {
            println '### TRAINING SET'
            assessCrossEntropyCostAndAccuracy(network, trainingImages)
        }
        println ''
    }

    private static void dumpNetwork(NeuralNetwork network) {
        network.dumpWeights(0)
        println ''
        network.dumpNodeDeltas()
        println ''
        network.dumpAccumulatedNodeDeltas()
        println ''
        network.dumpOutputValues()
        println ''
    }


    private
    static void trainMiniBatch(double[] inputTransfer, double[] softmaxTransfer, NeuralNetwork network, MnistImages images, int from, int miniBatchSize, double η_trainingRate, boolean doLog) {
        int to = from + miniBatchSize
        if (doLog) {
            println "TRAINING [$miniBatchSize $images.type], images:[$from, $to>"
        }
        long nanosStart = System.nanoTime()

        // Clear the accumulated node deltas, to prepare for new accumulation for minibatch
        for (int l = 1; l < network.layers.size(); l++) {
            Layer_WithInputs layer = (Layer_WithInputs) network.layers[l]
            layer.clearAccumulatedNodeDelta()
        }

        // :: Feed forward input values through the network, then accumulate node deltas in backpropagation
        for (int i = from; i < to; i++) {
            images.getImage(i, inputTransfer)

            // Do feed-forward of the input values through the output
            network.inputLayer.setInputs(inputTransfer)
            network.calculate()

            // Calculate the "error": i.e. find the node delta for the output layer. (We do not actually calculate the error function, only its derivative).
            int label = images.getLabel(i)
            network.outputLayer.softmaxAndCalculateCrossEntropyNodeDeltaForOutputLayerNeurons(softmaxTransfer, label)

            // Do backpropagation of node deltas for this sample
            for (int l = 1; l < network.layers.size() - 1; l++) {
                Layer_WithInputs layer = (Layer_WithInputs) network.layers[l]
                layer.backpropagateNodeDeltasFromNextLayerIntoThisLayer((Layer_WithInputs) network.layers[l + 1])
            }
            // .. then accumulate the node deltas into the accumulator
            for (int l = 1; l < network.layers.size(); l++) {
                Layer_WithInputs layer = (Layer_WithInputs) network.layers[l]
                layer.accumulateNodeDeltaForMiniBatch()
            }
        }
        // Apply the accumulated node deltas to the weights (and then clear the accumulator)
        for (int l = 1; l < network.layers.size(); l++) {
            Layer_WithInputs layer = (Layer_WithInputs) network.layers[l]
            layer.updateWeightsAndBiasWithMinibatchAccumulatedNodeDelta(η_trainingRate)
        }

        if (doLog) {
            println " Done, time: ${Static.timingFrom(nanosStart)}"
        }
    }

    private
    static void assessCrossEntropyCostAndAccuracy(NeuralNetwork network, MnistImages images) {
        println "Evaluating [$images.size $images.type]:"
        long nanosStart = System.nanoTime()
        double summedCrossEntropy = 0
        double[] softmaxOutputTransfer = new double[network.outputLayer.neurons.size()]
        int correct = 0
        double[] inputTransfer = new double[28 * 28]
        for (int i = 0; i < images.size; i++) {
            images.getImage(i, inputTransfer)
            network.inputLayer.setInputs(inputTransfer)
            network.calculate()
            int label = images.getLabel(i)
            summedCrossEntropy += network.outputLayer.softmaxAndSumCrossEntropyForOutputNeuronsVsTarget(softmaxOutputTransfer, label)
            int predictedLabel = network.outputLayer.getIdxOfNeuronMaxOutputValue()
            if (predictedLabel == label) {
                correct++
            }
        }

        println " Done, time: ${Static.timingFrom(nanosStart)}"
        println " Σsamples(crossEntropy): ${Static.ff(summedCrossEntropy)}"
        println " ACCURACY: $correct out of $images.size, which is ${(correct / (double) images.size) * 100}%"
    }

    private static void infoAboutSingleImage(NeuralNetwork network, MnistImages trainingImages, int imageNumber) {
        int label = trainingImages.getLabel(imageNumber)

        // Calculate for image imageNumber
        double[] inputTransfer = new double[28 * 28]
        trainingImages.getImage(imageNumber, inputTransfer)
        network.inputLayer.setInputs(inputTransfer)
        network.calculate()

        int outputLayerSize = network.outputLayer.neurons.size()

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

}
