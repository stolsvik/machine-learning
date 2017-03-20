package com.stolsvik.neurons.mnist;

import java.util.Random;

import com.stolsvik.neurons.mnist.MnistReader.MnistFile;

/**
 * @author Endre Stølsvik, http://endre.stolsvik.com, 2017-03-14 23:20
 */
public class MnistImages {

    private MnistFile _mnistFile;
    private byte[][] _images;
    private int[] _labels;

    private Random _random;

    private MnistImages(MnistFile mnistFile, byte[][] images, int[] labels) {
        _mnistFile = mnistFile;
        _images = images;
        _labels = labels;
        setRandomSeed(1l);
    }

    /**
     * @return the MnistFile which this {@link MnistImages} represents, i.e. TRAINING or TEST.
     */
    public MnistFile getType() {
        return _mnistFile;
    }

    public static MnistImages getTraining() {
        byte[][] images = MnistReader.readTrainingFile(MnistFile.TRAINING_IMAGES);
        int[] labels = MnistReader.readLabelsFile(MnistFile.TRAINING_LABELS);
        return new MnistImages(MnistFile.TRAINING_IMAGES, images, labels);
    }

    public static MnistImages getTest() {
        byte[][] images = MnistReader.readTrainingFile(MnistFile.TEST_IMAGES);
        int[] labels = MnistReader.readLabelsFile(MnistFile.TEST_LABELS);
        return new MnistImages(MnistFile.TEST_IMAGES, images, labels);
    }

    public MnistImages splitOffValidation(int numberOfImagesToSplitOff) {
        if (numberOfImagesToSplitOff > _images.length) {
            throw new IllegalArgumentException("Cannot split out more images from this instance that it has.");
        }
        int originalNumberOfImages = _images.length;
        byte[][] splitOffImages = new byte[numberOfImagesToSplitOff][];
        int[] splitOffLabels = new int[numberOfImagesToSplitOff];
        byte[][] retainedImages = new byte[originalNumberOfImages - numberOfImagesToSplitOff][];
        int[] retainedLabels = new int[originalNumberOfImages - numberOfImagesToSplitOff];
        for (int i = 0; i < originalNumberOfImages; i++) {
            if (i < numberOfImagesToSplitOff) {
                splitOffImages[i] = _images[i];
                splitOffLabels[i] = _labels[i];
            }
            else {
                retainedImages[i - numberOfImagesToSplitOff] = _images[i];
                retainedLabels[i - numberOfImagesToSplitOff] = _labels[i];
            }
        }
        _images = retainedImages;
        _labels = retainedLabels;
        return new MnistImages(_mnistFile, splitOffImages, splitOffLabels);
    }

    public void setRandomSeed(long seed) {
        _random = new Random(seed);
    }

    public void randomize() {
        for (int i = _images.length; i > 1; i--) {
            int j = _random.nextInt(i);
            int tempInt = _labels[i - 1];
            _labels[i - 1] = _labels[j];
            _labels[j] = tempInt;

            byte[] tempBArray = _images[i - 1];
            _images[i - 1] = _images[j];
            _images[j] = tempBArray;
        }
    }

    private static void swap(int[] array, int i, int j) {
    }

    public int getSize() {
        return _images.length;
    }

    public byte[] getImage(int idx) {
        return _images[idx];
    }

    public void getImage(int idx, double[] dst) {
        byte[] bytes = _images[idx];
        for (int i = 0; i < (28 * 28); i++) {
            dst[i] = bytes[i] & 0xff; // Values 0-255, not -128 to 127.
        }
    }

    public int getLabel(int idx) {
        return _labels[idx];
    }

    public void getOneHotLabel(int idx, double[] oneHot) {
        int value = _labels[idx];
        oneHot[0] = value == 0 ? 1 : 0;
        oneHot[1] = value == 1 ? 1 : 0;
        oneHot[2] = value == 2 ? 1 : 0;
        oneHot[3] = value == 3 ? 1 : 0;
        oneHot[4] = value == 4 ? 1 : 0;
        oneHot[5] = value == 5 ? 1 : 0;
        oneHot[6] = value == 6 ? 1 : 0;
        oneHot[7] = value == 7 ? 1 : 0;
        oneHot[8] = value == 8 ? 1 : 0;
        oneHot[9] = value == 9 ? 1 : 0;
    }
}
