package com.stolsvik.neurons

import groovy.transform.CompileStatic
import groovy.util.logging.Slf4j

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-13 23:31
 */
@CompileStatic
@Slf4j
class Neuron_Input implements Neuron {

    double outputValue

    @Override
    void calculate() {
        /* no-op */
    }
}
