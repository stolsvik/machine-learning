package com.stolsvik.neurons

import groovy.transform.CompileStatic
import groovy.util.logging.Slf4j

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-13 23:30
 */
@CompileStatic
interface Neuron {

    double getOutputValue()

    void calculate()

}
