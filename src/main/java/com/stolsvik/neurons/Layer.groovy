package com.stolsvik.neurons

import groovy.transform.CompileStatic;

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-15 21:30
 */
@CompileStatic
interface Layer {

    void calculate()

    Neuron[] getNeurons()
}
