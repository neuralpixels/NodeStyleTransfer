import * as tf from '@tensorflow/tfjs-core';
import {biasAdd} from './tsextra'
import {valueMap, wait} from './basic'

/*
VGG19

canvas width = 1792

input   = [244,244,  3] = [

conv1_1 = [224,224, 64] = tiles[  8 x 8 ] = height 1792
conv1_2 = [224,224, 64] = tiles[  8 x 8 ] = height 1792
pool1 =   [112,112, 64] = tiles[ 16 x 4 ] = height 448
Total canvas size = [1792, 4032]

conv2_1 = [112,112,128] = tiles[ 16 x 8 ] = height 896
conv2_2 = [112,112,128] = tiles[ 16 x 8 ] = height 896
pool2 =   [ 56, 56,128] = tiles[ 32 x 4 ] = height 224
Total canvas size = [1792, 2016]

conv3_1 = [ 56, 56,256] = tiles[ 32 x 8 ] = height 448
conv3_2 = [ 56, 56,256] = tiles[ 32 x 8 ] = height 448
conv3_3 = [ 56, 56,256] = tiles[ 32 x 8 ] = height 448
conv3_4 = [ 56, 56,256] = tiles[ 32 x 8 ] = height 448
pool3 =   [ 28, 28,256] = tiles[ 64 x 4 ] = height 112
Total canvas size = [1792, 1904]

conv4_1 = [ 28, 28,512] = tiles[ 64 x 8 ] = height 224
conv4_2 = [ 28, 28,512] = tiles[ 64 x 8 ] = height 224
conv4_3 = [ 28, 28,512] = tiles[ 64 x 8 ] = height 224
conv4_4 = [ 28, 28,512] = tiles[ 64 x 8 ] = height 224
pool4 =   [ 14, 14,512] = tiles[128 x 4 ] = height 56
Total canvas size = [1792, 952]

conv5_1 = [ 14, 14,512] = tiles[128 x 4 ] = height 56
conv5_2 = [ 14, 14,512] = tiles[128 x 4 ] = height 56
conv5_3 = [ 14, 14,512] = tiles[128 x 4 ] = height 56
conv5_4 = [ 14, 14,512] = tiles[128 x 4 ] = height 56
pool5 =   [  7,  7,512] = tiles[256 x 2 ] = height 14
Total canvas size = [1792, 231]

Absolute canvas size = [1792, 9135]
 */


const vgg19_layers = [
    'conv1_1', 'conv1_2', 'pool1',

    'conv2_1', 'conv2_2', 'pool2',

    'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'pool3',

    'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4',

    'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5'
];


export default class VGG19 {

    constructor(
        variables:object,
    ) {
        this.variables = variables;
    }
    public variables:object;

    async process() {
        let next = await this.getInput();
        console.log('inputShape', next.shape);
        for(let i = 0; i < vgg19_layers.length; i++){
            const layerName = vgg19_layers[i];
            if(layerName.includes('conv')){
                next = await this.conv2d(next, layerName);
            } else {
                next = await this.maxPool(next);
            }
        }
        tf.dispose(next);
    }

    async conv2d(inputs:tf.Tensor4D, name:string){
        const conv = tf.conv2d(inputs, this.getVar(`${name}_kernel`), [1, 1], 'same');
        tf.dispose(inputs);
        await this.renderBreak();
        const bias = biasAdd(conv, this.getVar(`${name}_bias`));
        tf.dispose(conv);
        await this.renderBreak();
        const relu = tf.relu(bias);
        tf.dispose(bias);
        await this.renderBreak();
        return relu;
    }

    async maxPool(inputs:tf.Tensor4D){
        const pool = tf.maxPool(inputs, [2, 2], [2, 2], 'same');
        tf.dispose(inputs);
        await this.renderBreak();
        return pool;
    }

    getVar(name:string) {
        let variable;
        if (name in this.variables) {
            variable =  this.variables[name];
        } else {
            throw(`Variable does not exist ${name}`)
        }
        return variable;
    }

    async getInput():Promise<tf.Tensor4D>{
        // get image from canvas as float32 tensor with expanded dims for batch of 1
        const canvas = document.getElementById(`main-canvas`);
        const raw = tf.fromPixels(canvas);
        await this.renderBreak();
        const raw_float_not_expanded = tf.cast(raw, 'float32');
        tf.dispose(raw);
        const raw_float = tf.expandDims(raw_float_not_expanded, 0);
        tf.dispose(raw_float_not_expanded);
        await this.renderBreak();

        // transpose, convert to BGR color space, and adjust to mean pixel
        const vggMean = [
            tf.scalar(103.939, 'float32'),
            tf.scalar(116.779, 'float32'),
            tf.scalar(123.68, 'float32')
        ];
        const [r, g, b] = tf.split(raw_float, 3, 3);
        const r_shift = tf.sub(r, vggMean[2]);
        const b_shift = tf.sub(b, vggMean[0]);
        const g_shift = tf.sub(g, vggMean[1]);
        tf.dispose([r, g, b]);
        tf.dispose(vggMean);
        await this.renderBreak();
        const vgg_input = tf.concat([b_shift, g_shift, r_shift], 3);
        tf.dispose([r_shift, g_shift, b_shift]);
        return vgg_input;
        /*
        input_tensor = tf.clip_by_value(input_tensor, 0, 255)
        r, g, b = tf.split(axis=-1, num_or_size_splits=3, value=input_tensor)
        VGG_MEAN = [103.939, 116.779, 123.68]
        inputs = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=-1)
         */
    }


    async renderBreak(delay = 1) {
        await tf.nextFrame();
        if (delay > 0) {
            await wait(delay);
        }
    }
}