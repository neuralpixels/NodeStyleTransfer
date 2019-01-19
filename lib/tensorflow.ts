import {isNode} from './env'
import * as tf from '@tensorflow/tfjs-node-gpu';
if(!isNode){
    throw "Only NodeJS is currently supported"
}
export default tf;