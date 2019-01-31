import {isNode} from './env'
import * as tf from '@tensorflow/tfjs-core';
if(isNode){
    require('@tensorflow/tfjs-node-gpu')
}

export default tf;
