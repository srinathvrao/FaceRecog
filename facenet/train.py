import tensorflow as tf
from triplet_loss import triplet_loss
from fr_utils import *
from inception_blocks_v2 import *

def main():
    model = faceRecoModel(input_shape=(3,96,96))
    model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    model.save('trained_weights.h5')

    print_summary(model)

if __name__ == '__main__':
    main()
