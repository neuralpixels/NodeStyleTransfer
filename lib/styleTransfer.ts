import {loadRemoteVariablesPromise} from './modelLoader';
import tf from "./tensorflow";
import VGG19 from './VGG19';
import {getImageAsTensor, saveTensorAsImage} from './image';
import {IVariables} from './modelLoader'
import {valueMap, exponent, rjust} from './basic';
import {isNode} from './env'
import AggressiveOptimizer from './aggressive_optimizer';
import * as numeral from 'numeral';

const defaultCallback = (progressPercent, progressMessage, isProcessing = true): void => {
    console.log('NeuralNetRunner', {
        loadingPercent: progressPercent,
        loadingMessage: progressMessage,
        isLoading: isProcessing
    })
};

interface Size {
    width:number
    height:number
}

let modelPath;
if(isNode){
    modelPath = require('path').join(__dirname, '../weights/vgg19')
} else {
    modelPath = 'https://neuralpixels.com/wp-content/plugins/np-visualize-vgg19/model'
}

export class StyleTransfer {
    constructor(
        parameters: {
            content: string,
            style: string,
            output: string,
            iterations?: number,
            statusCallback?: object ,
            silent?:boolean,
            processTiled?:boolean
        }
    ) {
        let {
            content,
            style,
            output,
            iterations = 50,
            statusCallback = defaultCallback,
            silent = false,
            processTiled=true
        } = parameters;
        this.modelPath = modelPath;
        this.content = content;
        this.style = style;
        this.output = output;
        this.statusCallback = statusCallback;
        this.displayName = 'StyleTransfer';
        this.net = null;
        this.variables = {};
        this.manifest = null;
        this.vgg19 = null;
        this.isProcessing = true;
        this.pad = 20;
        this.learningRate = 25.5;
        this.iterations = iterations;
        this.iteration = 0;
        this.optimizer = tf.train.adam(this.learningRate);
        // this.optimizer = new AggressiveOptimizer();

        this.styleWeight = 7.5e-1;
        this.contentWeight = 1e0;

        this.contentLayers = null;
        this.styleLayers = null;
        this.outputImageLayers = null;
        this.silent = silent;
        this.processTiled = processTiled;
        this.tileSize = 128;
        this.tilePad = 8;
        this.paddings = [[0, 0], [this.pad, this.pad], [this.pad, this.pad], [0, 0]];
    }

    public modelPath: string;
    public content: string;
    public style: string;
    public output: string;
    public statusCallback: object;
    public displayName: string;
    public net: object | null;
    public variables: IVariables;
    public manifest: object;
    public vgg19: VGG19 | null;
    public isProcessing: boolean;
    public pad: number;
    public learningRate: number;
    public optimizer: tf.Optimizer|AggressiveOptimizer;
    public outputImage: tf.Variable;
    public contentLayers: any;
    public styleLayers: any;
    public outputImageLayers: any;
    public iterations: number;
    public iteration: number;
    public startSize: Size;
    public endSize: Size;
    public styleWeight: number;
    public contentWeight: number;
    public contentTensor: any;
    public silent: boolean;
    public processTiled: boolean;
    public tileSize: number;
    public tilePad: number;
    public paddings: [[number, number], [number, number], [number, number], [number, number]];

    finalCleanup() {
        // //tf.dispose(this.variables);
        //tf.dispose(this.contentLayers);
        //tf.dispose(this.styleLayers);
        //tf.dispose(this.outputImage);
        //tf.disposeVariables();
    }

    async initialize() {
        // todo update status
        try {
            this.variables = await loadRemoteVariablesPromise(this.modelPath);
            this.vgg19 = new VGG19(this.variables);
            return true;
        } catch (e) {
            return false;
        }
    }

    _seperated_l2_loss(y_pred, y_true) {
        return tf.tidy(()=>{
            let sum_axis = [1, 2];
            let _size = y_pred.shape[1] * y_pred.shape[2];
            if (y_pred.shape.length === 4) {
                sum_axis = [1, 2, 3];
                _size = y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3];
            }
            const diff = tf.sub(y_pred, y_true);
            const abs_diff = tf.abs(diff);
            const square_abs_diff = tf.square(abs_diff);
            const reduced_abs_diff = tf.sum(square_abs_diff, sum_axis);
            const two = tf.scalar(2.0, 'float32');
            const l2 = tf.div(reduced_abs_diff, two);
            const l2t2 = tf.mul(two, l2);
            const size = tf.scalar(_size, 'float32');
            return tf.div(l2t2, size);
        });
    }

    mergeTileLayers(inputsArr){
        return tf.tidy(()=>{
            let outputLayers = {};
            let numTiles = tf.scalar(inputsArr.length, 'float32');
            for(let i = 0; i < inputsArr.length; i++){
                for(let layerName in inputsArr[i]){
                    const layerVal = tf.div(inputsArr[i][layerName], numTiles);
                    if(!(layerName in outputLayers)){
                        outputLayers[layerName] = layerVal;
                    } else {
                        const newOutput = tf.add(outputLayers[layerName], layerVal);
                        // tf.dispose([outputLayers[layerName], layerVal]);
                        outputLayers[layerName] = newOutput;
                    }
                }
            }
            return outputLayers;
        });

    }

    _styleLoss(tileNum) {
        return tf.tidy(()=>{
            let loss = tf.scalar(0.0);
            let next = loss;
            // add the output layers to the tmp object
            const tileLayers = tf.tidy(()=>{
                let vggIn = this.getTiles(this.outputImage, tileNum);
                const vggInput = this.vgg19.prepareInput(vggIn);
                return this.vgg19.getLayers(vggInput, 'style', true, true);
            });
            let outputImageLayersTmp = [];
            for(let i=0; i < this.outputImageLayers.length; i++){
                if(i === tileNum){
                    outputImageLayersTmp.push(tileLayers)
                } else{
                    outputImageLayersTmp.push(this.outputImageLayers[i])
                }
            }
            const outputGrams = this.mergeTileLayers(outputImageLayersTmp);
            for (let i = 0; i < this.vgg19.vgg19_style_layers.length; i++) {
                loss = tf.tidy(()=>{
                    const layerName = this.vgg19.vgg19_style_layers[i];
                    const outputImageGrams = outputGrams[layerName];
                    const styleGrams = this.styleLayers[layerName];
                    const rawLayerLoss = this._seperated_l2_loss(
                        outputImageGrams,
                        styleGrams
                    );
                    //tf.dispose([styleGrams, outputImageGrams]);
                    const layerLoss = tf.mean(rawLayerLoss);
                    //tf.dispose(rawLayerLoss);
                    const layerWeight = tf.scalar(this.vgg19.vgg19_layer_weights[layerName]['content'], 'float32');
                    const weightedLoss = tf.mul(layerLoss, layerWeight);
                    //tf.dispose([layerLoss, layerWeight]);
                    next = tf.add(weightedLoss, loss);
                    tf.dispose(loss);
                    return next;
                });
            }
            const numLayers = tf.scalar(this.vgg19.vgg19_style_layers.length, 'float32');
            const avgLayerLoss = tf.div(loss, numLayers);
            //tf.dispose([numLayers, loss]);
            const styleWeight = tf.scalar(this.styleWeight, 'float32');
            const lossOutput = tf.mul(avgLayerLoss, styleWeight);
            //tf.dispose([avgLayerLoss, styleWeight]);
            return lossOutput;
        });
    }

    loss = (tileNum=0) => {
        return tf.tidy(()=>{
            const styleLoss = this._styleLoss(tileNum);
            // const contentLoss = this._contentLoss();
            // const loss = tf.add(contentLoss, styleLoss);
            const lossScalar = styleLoss.asScalar();
            return lossScalar;
        });
    };


    async process(inputs:tf.Tensor, iterations: number=100) {
        this.contentTensor = tf.tidy(()=>{
            let inputsF32;
            if(inputs.dtype !== 'float32'){
                inputsF32 = tf.cast(inputs, 'float32');
                tf.dispose(inputs);
            } else {
                inputsF32 = inputs;
            }
            if(this.processTiled){
                this.updatePaddings(inputsF32);
            }
            return tf.pad4d(inputsF32, this.paddings)
        });

        this.startSize = {
            width:Math.floor(this.contentTensor.shape[1] * 0.5),
            height:Math.floor(this.contentTensor.shape[2] * 0.5)
        };
        this.endSize = {
            width:this.contentTensor.shape[1],
            height:this.contentTensor.shape[2]
        };
        // @ts-ignore
        const scaledContent = tf.image.resizeBilinear(this.contentTensor,[this.startSize.height, this.startSize.width]);
        tf.dispose(this.outputImage);
        this.outputImage = tf.variable(scaledContent, true, 'output');
        tf.dispose(scaledContent);

        this.iterations = iterations;
        this.iteration = 0;
        // await this.preComputeContent();
        await this.preComputeStyle();
        // this.preComputeContent();
        for (let i = 0; i < iterations; i++) {
            await this.runIteration();
            this.iteration++;
        }
        return tf.tidy(()=>{
            let slice_begin = [
                0,
                this.paddings[1][0],
                this.paddings[2][0],
                0
            ];
            let slice_size = [
                1,
                Math.floor(this.outputImage.shape[1]) - (this.paddings[1][0] + this.paddings[1][1]),
                Math.floor(this.outputImage.shape[2]) - (this.paddings[2][0] + this.paddings[2][1]),
                3
            ];
            return tf.slice(this.outputImage,
                slice_begin,
                slice_size
            )
        });
        // this.finalCleanup();
    }

    async preComputeStyle() {
        // const content = tf.fromPixels(document.getElementById('style-canvas'));
        const style = await getImageAsTensor(this.style);
        const vggInput = this.vgg19.prepareInput(style);
        tf.dispose([style, this.styleLayers]);
        this.styleLayers = this.vgg19.getLayers(vggInput, 'style', true);
        tf.dispose([style, vggInput]);
    }

    computeOutputImage() {
        tf.dispose(this.outputImageLayers);
        this.outputImageLayers = tf.tidy(()=>{
            const tiles = this.getTiles(this.outputImage);
            let imageLayers = [];
            for(let i =0; i < tiles.length; i++) {
                const tileLayers = tf.tidy(()=>{
                    let vggIn = tiles[i];
                    const vggInput = this.vgg19.prepareInput(vggIn);
                    return this.vgg19.getLayers(vggInput, 'style', true, this.processTiled);
                });
                imageLayers.push(tileLayers);
            }
            return imageLayers;

        });
    }

    runIteration() {
        let size:Size = {
            width:this.outputImage.shape[2],
            height:this.outputImage.shape[1]
        };
        if (this.iteration === Math.floor(this.iterations / 2)) {
            size = {
                width: this.endSize.width,
                height: this.endSize.height
            };
            const newVar = tf.tidy(() => {
                let slice_begin = [
                    0,
                    this.paddings[1][0],
                    this.paddings[2][0],
                    0
                ];
                let slice_size = [
                    1,
                    Math.floor(this.outputImage.shape[1]) - (this.paddings[1][0] + this.paddings[1][1]),
                    Math.floor(this.outputImage.shape[2]) - (this.paddings[2][0] + this.paddings[2][1]),
                    3
                ];
                const croppedOutput =  tf.slice(this.outputImage,
                    slice_begin,
                    slice_size
                );
                //  @ts-ignore
                const newV =  tf.variable(tf.image.resizeBilinear(croppedOutput, [size.height, size.width]));
                if(this.processTiled){
                    this.updatePaddings(newV);
                }
                // @ts-ignore
                return tf.pad4d(newV, this.paddings)

            });
            tf.dispose(this.outputImage);

            this.outputImage = tf.variable(newVar);
            tf.dispose(newVar);
            if(this.optimizer instanceof AggressiveOptimizer){
                try{
                    this.optimizer.reset();
                } catch(e){

                }
            }
        }

        const [batch, rows, cols, chan] = this.outputImage.shape;
        let numTiles = 1;
        if (this.processTiled) {
            // padding should be already added for even splits
            let num_row_splits = Math.floor(rows / this.tileSize);
            let num_col_splits = Math.floor(cols / this.tileSize);
            numTiles = num_col_splits * num_row_splits;
        }
        let grad = tf.zerosLike(this.outputImage);
        let loss = tf.scalar(0.0, 'float32');
        for(let tileNum = 0; tileNum < numTiles; tileNum++){
            // compute all tiles
            this.computeOutputImage();
            const tileGrad = tf.tidy(()=>{
                const tileLoss = () =>{
                  return this.loss(tileNum);
                };
                const {value, grads} = this.optimizer.computeGradients(
                    tileLoss,
                    [this.outputImage]
                );
                const newLoss = tf.add(loss, tf.div(value, numTiles));
                tf.dispose(loss);

                // @ts-ignore
                loss = newLoss;
                tf.keep(loss);
                const varName = Object.keys(grads)[0];
                return grads[varName];
            });
            const newGrad = tf.add(grad, tileGrad);
            tf.dispose([grad, tileGrad]);
            // @ts-ignore
            grad = newGrad;
        }

        // this.optimizer.applyGradients({
        //     [this.outputImage.name]: grad
        // }, loss);
        this.optimizer.applyGradients({
            [this.outputImage.name]: grad
        });
        const memory = tf.memory();
        //tf.dispose(styleLoss);
        let printArr = [
            `[${rjust(this.iteration + 1, `${this.iterations}`.length)}/${this.iterations}]`,
            `loss: ${exponent(loss.dataSync(), 2)}`,
            `numTiles: ${numTiles}`,
            `numTensors: ${memory.numTensors}`,
            `numBytes: ${numeral(memory.numBytes).format('0,0')}`
        ];
        if(!this.silent){
            console.log(printArr.join(' '));
        }
        tf.dispose([loss, grad])
    }

    updatePaddings(inputs:tf.Tensor){
        const [batch, rows, cols, chan] = inputs.shape;
        let rows_pad = rows + (this.tilePad * 2);
        let cols_pad = cols + (this.tilePad * 2);
        let row_pad = this.tileSize - (rows_pad % this.tileSize);
        let col_pad = this.tileSize - (cols_pad % this.tileSize);
        if(row_pad === this.tileSize){
            row_pad = 0;
        }
        if(col_pad === this.tileSize) {
            col_pad = 0;
        }
        this.paddings = [
            [0, 0],
            [this.tilePad + Math.floor(row_pad / 2), this.tilePad + Math.ceil(row_pad / 2)],
            [this.tilePad + Math.floor(col_pad / 2), this.tilePad + Math.ceil(col_pad / 2)],
            [0, 0]
        ];
    }

    getTiles(inputs, tileNum:number|null=null):tf.Tensor[] {
        const inputsShape = inputs.shape;
        return tf.tidy(()=>{
            const [rows, cols] = inputsShape.slice(1, 3);
            let tiles = [];
            if (!this.processTiled) {
                tiles = [inputs]
            } else {
                // padding should be already added for even splits
                let num_row_splits = Math.floor(rows / this.tileSize);
                let num_col_splits = Math.floor(cols / this.tileSize);

                tiles = [];
                let n = 0;
                for(let c = 0; c < num_col_splits; c++){
                    for(let r = 0; r < num_row_splits; r++){
                        if(tileNum == null ||  n == tileNum){
                            let slice_row_start = r * this.tileSize;
                            let slice_col_start = c * this.tileSize;
                            let slice_begin = [
                                0, slice_row_start, slice_col_start, 0
                            ];
                            let slice_row = this.tileSize + (this.tilePad * 2);
                            if(slice_row + slice_row_start > rows) {
                                slice_row = rows - slice_row_start
                            }
                            let slice_col = this.tileSize + (this.tilePad * 2);
                            if(slice_col + slice_col_start > cols) {
                                slice_col = cols - slice_col_start;
                            }

                            let slice_size = [
                                1, slice_row, slice_col, 3
                            ];
                            let slice = tf.slice(
                                inputs,
                                slice_begin,
                                slice_size
                            );
                            tiles.push(
                                tf.pad(
                                    slice,
                                    [
                                        [0, 0],
                                        [this.tilePad, this.tilePad],
                                        [this.tilePad, this.tilePad],
                                        [0, 0]
                                    ]
                                )
                            )
                        }
                        n++;
                    }
                }
            }
            if(tileNum === null) {
                return tiles;
            } else {
                return tiles[0];
            }

        });
    }
}