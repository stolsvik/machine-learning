package com.stolsvik.neurons

import groovy.transform.CompileStatic
import org.slf4j.Logger

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-01-21 22:22
 */
@CompileStatic
class Static {

    static double VALUE_WHEN_ZERO = Math.exp(-100d)

    static String timingFrom(long fromNanos) {
        long now = System.nanoTime()
        return timingBetween(fromNanos, now)
    }

    static String timingBetween(long fromNanos, long toNanos) {
        double millisBetween = (toNanos - fromNanos) / 1_000_000d
        return sprintf("%.3f ms", millisBetween)
    }

    static ff(float f) {
        if (f == Float.MAX_VALUE) {
            return 'MAXF'
        }
        ff((double) f)
    }

    static ff(double d) {
        if (d == Double.MAX_VALUE) {
            return 'MAXD'
        }
        sprintf("%.8f", d)
    }

    static <T> T time(Logger logTo, String explain, Closure c) {
        if (!logTo.isDebugEnabled()) {
            return c()
        }
        long startNanos = System.nanoTime()
        T ret = c()
        String time = timingFrom(startNanos)
        logTo.debug "$explain: $time"
        return ret
    }

    static void parallelize(Logger log, String what, String threadName = null, int startLine, int numLines, Parallel parallel) {
        int numSplits = Runtime.getRuntime().availableProcessors()
        time(log, "$what, parallelized on $numSplits threads") {
            Thread[] threads = new Thread[numSplits]
            int lastTo = startLine
            // :: Fire up threads
            for (int i = 0; i < numSplits; i++) {
                int fromLine = lastTo
                int toLine = (int) Math.round((numLines / (double) numSplits) * (double) (i + 1))
                lastTo = toLine
                String actualThreadName = (threadName != null ? "$threadName #$i" : "$what #$i")
                threads[i] = new Thread({
                    parallel.runLines(fromLine, toLine)
                }, actualThreadName)
                threads[i].start()
            }
            // :: Join threads.
            for (int i = 0; i < numSplits; i++) {
                threads[i].join()
            }
        }
    }

    @FunctionalInterface
    interface Parallel {
        void runLines(int fromLine, int toLine)
    }
}
