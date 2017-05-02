package com.stolsvik.neurons

import com.stolsvik.neurons.mnist.MnistImages
import groovy.transform.CompileStatic
import java.util.function.DoubleSupplier

/**
 * Backprop: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example
 *
 * http://www.deeplearningbook.org/
 *
 * https://www.reddit.com/r/MachineLearning/comments/45gkoy/has_anyone_read_this_neural_networks_and_deep/
 *
 * http://neuralnetworksanddeeplearning.com/
 *
 * https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
 *
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-14 00:16
 */
@CompileStatic
class Mnist {

    static void main(String... args) {
        NeuralNetwork network = new NeuralNetwork()
        network.addLayer(Layer_Input.createInputLayer(28 * 28))
//        network.addLayer(Layer_WithInputs.createSigmoidLayer(network.layers.size(), 250, network.lastLayer.neurons))
        network.addLayer(Layer_WithInputs.createLeakyReLULayer(network.layers.size(), 400, 0.1d, network.lastLayer.neurons))
        network.addLayer(Layer_WithInputs.createLeakyReLULayer(network.layers.size(), 250, 0.1d, network.lastLayer.neurons))
        network.addLayer(Layer_WithInputs.createLinearLayer(network.layers.size(), 10, network.lastLayer.neurons))

        Random random = new Random(2l)
        network.initialize({ return random.nextGaussian() * 0.1d } as DoubleSupplier,
                { return random.nextGaussian() * 0.1d } as DoubleSupplier)

        MnistImages trainingImages = MnistImages.training
        MnistImages testImages = MnistImages.test

        println "\n========= INITIAL VALUES, before any training ========="
        network.infoAboutSingleImage(network, trainingImages, 0)
        network.dumpNetwork(network)

        network.assessCrossEntropyCostAndAccuracy(network, testImages)
        // assessCrossEntropyCostAndAccuracy(network, trainingImages)

        network.train(trainingImages, testImages, 10, 0.001d)
    }
}
