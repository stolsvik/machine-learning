package com.stolsvik.neurons

import groovy.transform.CompileStatic

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-25 20:51
 */
@CompileStatic
abstract class Layer_Abstract implements Layer {

    int layerIdx

    Layer_Abstract(int layerIdx) {
        this.layerIdx = layerIdx
    }

    @Override
    String toString() {
        "${this.class.simpleName}.neurons.outputValue:\n${dumpCurrentOutputValues()}"
    }

    String dumpCurrentOutputValues() {
        StringBuilder buf = new StringBuilder()
        neurons.eachWithIndex { Neuron neuron, int i ->
            Static.singleDumpElement(buf, i, Static.ff4(neuron.outputValue))
        }
        buf.substring(0, buf.length() - 1)
    }
}
