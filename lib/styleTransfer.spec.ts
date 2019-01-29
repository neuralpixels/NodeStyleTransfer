import 'mocha';
import { expect } from 'chai';
import * as path from 'path';
process.env['TF_CPP_MIN_LOG_LEVEL'] = '2';
process.env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
process.env["CUDA_VISIBLE_DEVICES"] = `0`;
import tf from './tensorflow';
import {StyleTransfer} from './styleTransfer';
import {getImageAsTensor} from './image';

let mochaAsync = (fn) => {
    return (done) => {
        fn.call().then(done(), (err)=>{done(err)});
    };
};

describe('styleTransfer memory cleanup', () => {
    it('should remove all tensors from memory', mochaAsync(async () => {
        let args = {
            content: path.join(__dirname, '../test/lenna_512.jpg'),
            style: path.join(__dirname, '../test/lenna_512_paint01.jpg'),
            output: path.join(__dirname, '../test/paint01.jpg'),
            silent: true
        };
        const styleTransfer = new StyleTransfer(args);
        await styleTransfer.initialize();
        const startMemory = tf.memory();
        const contentTensor = await getImageAsTensor(args.content, true, 'float32');
        const outputTensor = await styleTransfer.process(contentTensor, 2);
        tf.dispose(outputTensor);
        tf.dispose(styleTransfer.vgg19.variables);

        const endMemory = tf.memory();
        const tensorDiff = endMemory.numTensors - startMemory.numTensors;
        expect(tensorDiff).to.equal(0);
    }));
});