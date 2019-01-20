import {loadRemoteVariablesPromise} from './modelLoader';
import tf from "./tensorflow";
import VGG19 from './VGG19';
import {getImageAsTensor, saveTensorAsImage} from './image';
import {IVariables} from './modelLoader'
import {valueMap, exponent, rjust} from './basic';
import {isNode} from './env'

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
        parameters: { content: string, style: string, output: string, iterations?: number, statusCallback?: object }
    ) {
        let {content, style, output, iterations = 100, statusCallback = defaultCallback} = parameters;
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
        this.pad = 10;
        this.learningRate = 25.5;
        this.iterations = iterations;
        this.iteration = 0;
        this.optimizer = tf.train.adam(this.learningRate);

        this.styleWeight = 7.5e-1;
        this.contentWeight = 1e0;

        this.contentLayers = null;
        this.styleLayers = null;
        this.outputImageLayers = null;
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
    public optimizer: tf.Optimizer;
    public outputImage: any;
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
            // this.config = await updateBuildConfig(this.config);
            this.variables = await loadRemoteVariablesPromise(this.modelPath);
            this.vgg19 = new VGG19(this.variables);
            const contentTensorInt = await getImageAsTensor(this.content, true);
            this.contentTensor = tf.cast(contentTensorInt, 'float32');
            this.startSize = {
                width:Math.floor(this.contentTensor.shape[1] * 0.8),
                height:Math.floor(this.contentTensor.shape[2] * 0.8)
            };
            this.endSize = {
                width:this.contentTensor.shape[1],
                height:this.contentTensor.shape[2]
            };
            // @ts-ignore
            const scaledContent = tf.image.resizeBilinear(this.contentTensor,[this.startSize.height, this.startSize.width]);
            this.outputImage = tf.variable(scaledContent, true, 'output');
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

    _convert_to_gram_matrix(inputs) {
        return tf.tidy(()=>{
            const [batch, height, width, filters] = inputs.shape;
            const feats = tf.reshape(inputs, [batch, height * width, filters]);
            const feats_t = tf.transpose(feats, [0, 2, 1]);
            const grams_raw = tf.matMul(feats_t, feats);
            const size = tf.scalar(height * width * filters, 'float32');
            return tf.div(grams_raw, size);
        });
    }

    _styleLoss() {
        return tf.tidy(()=>{
            let loss = tf.scalar(0.0);
            let next = loss;
            for (let i = 0; i < this.vgg19.vgg19_style_layers.length; i++) {
                const layerName = this.vgg19.vgg19_style_layers[i];
                const outputImageGrams = this._convert_to_gram_matrix(this.outputImageLayers[layerName]);
                const styleGrams = this._convert_to_gram_matrix(this.styleLayers[layerName]);
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
                //tf.dispose(loss);
                loss = next;
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

    loss = () => {
        return tf.tidy(()=>{
            this.computeOutputImage();
            const styleLoss = this._styleLoss();
            // const contentLoss = this._contentLoss();
            // const loss = tf.add(contentLoss, styleLoss);
            const lossScalar = styleLoss.asScalar();
            //tf.dispose(styleLoss);
            let printArr = [
                `[${rjust(this.iteration + 1, `${this.iterations}`.length)}/${this.iterations}]`,
                `loss: ${exponent(lossScalar.dataSync(), 2)}`,
                `style: ${exponent(styleLoss.dataSync(), 2)}`,
            ];
            console.log(printArr.join(' '));
            return lossScalar;
        });
    };


    async process(iterations: number) {
        this.iterations = iterations;
        this.iteration = 0;
        // await this.preComputeContent();
        await this.preComputeStyle();
        // this.preComputeContent();
        for (let i = 0; i < iterations; i++) {
            await this.runIteration();
            this.iteration++;
        }
        await saveTensorAsImage(this.outputImage.clone(), this.output)
        // this.finalCleanup();
    }

    async preComputeStyle() {
        // const content = tf.fromPixels(document.getElementById('style-canvas'));
        const style = await getImageAsTensor(this.style);
        const vggInput = this.vgg19.prepareInput(style);
        this.styleLayers = this.vgg19.getLayers(vggInput, 'style');
    }

    computeOutputImage() {
        let vggIn = this.outputImage;
        const vggInput = this.vgg19.prepareInput(vggIn);
        this.outputImageLayers = this.vgg19.getLayers(vggInput, 'both');
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
                const newVarScaled = tf.variable(tf.image.resizeBilinear(this.outputImage, [size.height, size.width]));
                const contentScalar = tf.scalar(0.1);
                const varScalar = tf.sub(tf.scalar(1.0), contentScalar);
                const scaledContent = tf.variable(tf.image.resizeBilinear(this.contentTensor, [size.height, size.width]));
                const contentAdd = tf.mul(scaledContent, contentScalar);
                const varAdd = tf.mul(newVarScaled, varScalar);
                return tf.add(varAdd, contentAdd)
            });
            tf.dispose(this.outputImage);

            this.outputImage = tf.variable(newVar);
        }

        this.optimizer.minimize(this.loss, false, [this.outputImage]);
    }
}