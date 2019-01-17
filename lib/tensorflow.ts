const isNode = Object.prototype.toString.call(typeof process !== 'undefined' ? process : 0) === '[object process]';
import * as tf from '@tensorflow/tfjs-node-gpu';
if(!isNode){
    throw "Only NodeJS is currently supported"
}
export default tf;