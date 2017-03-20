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
//        network.addLayer(Layer_WithInputs.createSigmoidLayer(500, network.lastLayer.neurons))
        network.addLayer(Layer_WithInputs.createLinearLayer(10, network.lastLayer.neurons))

        Random random = new Random()
        random.setSeed(2l)
        network.initialize({ return random.nextGaussian() * 0.1d } as DoubleSupplier,
                { return random.nextGaussian() * 0.1d } as DoubleSupplier)

        MnistImages trainingImages = MnistImages.training
        MnistImages testImages = MnistImages.test

        infoAboutSingleImage(network, trainingImages, 0)

        boolean doLog = false

        assessCrossEntropyCostAndAccuracy(network, testImages)
        assessCrossEntropyCostAndAccuracy(network, trainingImages)

        println '\n========= TRAINING =========='

        int miniBatchSize = 1
        int batches = (int) (trainingImages.size / miniBatchSize)
//        trainMiniBatch(network, trainingImages, 36000, 100)
//        trainMiniBatch(network, trainingImages, 36100, 100)
        for (int epoch = 0; epoch < 10000; epoch++) {
            println "TRAINING Epoch $epoch"
            long startEpochNanos = System.nanoTime()
            trainingImages.randomize()
            for (int i = 0; i < batches; i++) {
                trainMiniBatch(network, trainingImages, i * miniBatchSize, miniBatchSize, 0.01d, doLog)
            }
            println "  \\- Epoch $epoch done, took ${Static.timingFrom(startEpochNanos)}."
            infoAboutSingleImage(network, trainingImages, 0)
            assessCrossEntropyCostAndAccuracy(network, testImages)
            if (epoch % 10 == 0) {
                println '### TRAINING SET'
                assessCrossEntropyCostAndAccuracy(network, trainingImages)
            }
            println ''
        }
        assessCrossEntropyCostAndAccuracy(network, trainingImages)
    }

    private
    static void trainMiniBatch(NeuralNetwork network, MnistImages images, int from, int size, double trainingRate, boolean doLog) {
        double[] inputTransfer = new double[28 * 28]
        int to = from + size
        if (doLog) {
            println "TRAINING [size $images.type], images:[$from, $to>"
        }
        long nanosStart = System.nanoTime()
        double[] outputTransfer = new double[network.outputLayer.neurons.size()]
        // Sum up cross entropy
        for (int i = from; i < to; i++) {
            images.getImage(i, inputTransfer)
            network.inputLayer.setInputs(inputTransfer)
            network.calculate()
            int label = images.getLabel(i)
            network.outputLayer.softmaxAndBackpropagateErrors(outputTransfer, label)
        }

        if (doLog) {
            println " Done, time: ${Static.timingFrom(nanosStart)}"
        }
    }

    private
    static void assessCrossEntropyCostAndAccuracy(NeuralNetwork network, MnistImages images) {
        double[] inputTransfer = new double[28 * 28]
        println "Evaluating [$images.size $images.type]:"
        long nanosStart = System.nanoTime()
        double summedCrossEntropy = 0
        double[] softmaxOutputTransfer = new double[network.outputLayer.neurons.size()]
        int correct = 0
        for (int i = 0; i < images.size; i++) {
            images.getImage(i, inputTransfer)
            network.inputLayer.setInputs(inputTransfer)
            network.calculate()
            int label = images.getLabel(i)
            summedCrossEntropy += network.outputLayer.softmaxAndSumCrossEntropy(softmaxOutputTransfer, label)
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

        double[] softmaxOutputTransfer = new double[outputLayerSize]
        // Doing this here, since we'll then do the softmax:
        double sparseCrossEntropy = network.outputLayer.softmaxAndSumCrossEntropy(softmaxOutputTransfer, label)
        double sum = 0
        softmaxOutputTransfer.eachWithIndex { double value, int i ->
            Neuron_WithInputs neuron = (Neuron_WithInputs) network.outputLayer.neurons[i]
            double ln = Layer_WithInputs.crossEntropyOfSingle(value, label == i)
            sum += ln
            println "  Neuron #$i: t: ${(int) oneHotLabelTransfer[i]}, a: ${Static.ff(neuron.outputValue)}," +
                    " softmax(a): ${Static.ff(value)}, crossEntropy: ${Static.ff(ln)}," +
                    " δ_nodeDelta:${Static.ff(neuron.δ_nodeDelta)} # rawValue:${neuron.calculateRawOutputValue()}"
        }
        println "  Σneurons(crossEntropy): ${sum}"
        println "  vs. sparse cross entropy: $sparseCrossEntropy"
    }

}
