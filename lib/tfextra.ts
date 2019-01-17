import * as tf from '@tensorflow/tfjs-node-gpu';


export function biasAdd(
    x:tf.Tensor4D,
    bias:tf.Tensor1D | tf.Tensor3D
){
    return tf.tidy(() => {
        if (bias.rank !== 1 && bias.rank !== x.rank) {
            throw (
                'Unexpected bias dimensions: ' + bias.rank +
                '; expected it to be 1 or ' + x.rank);
        }
        const biasShape = bias.shape;

        let y;
        if (biasShape.length === 1) {
            y = x.add(bias.reshape([1, 1, 1, biasShape[0]]));
        } else {
            y = x.add(bias.reshape([1].concat(biasShape)));
        }

        tf.dispose(x);
        return y;
    });
}