import {loadRemoteVariablesPromise} from './modelLoader';
import tf from "./tensorflow";
import VGG19 from './VGG19';
import {getImageAsTensor, saveTensorAsImage} from './image';
import {IVariables} from './modelLoader'
import {valueMap} from './basic'

const defaultCallback = (progressPercent, progressMessage, isProcessing = true): void => {
    console.log('NeuralNetRunner', {
        loadingPercent: progressPercent,
        loadingMessage: progressMessage,
        isLoading: isProcessing
    })
};

export class StyleTransfer {
    constructor(
        parameters: { content: string, style: string, output: string, iterations?: number, statusCallback?: object }
    ) {
        let {content, style, output, iterations = 1000, statusCallback = defaultCallback} = parameters;
        this.modelPath = 'https://neuralpixels.com/wp-content/plugins/np-visualize-vgg19/model';
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
        this.learningRate = 2.55;
        this.iterations = iterations;
        this.iteration = 0;
        this.optimizer = tf.train.adam(this.learningRate);

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
    public optimizer: any;
    public outputImage: any;
    public contentLayers: any;
    public styleLayers: any;
    public outputImageLayers: any;
    public iterations: number;
    public iteration: number;

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
            const contentTensor = tf.cast(contentTensorInt, 'float32');
            const scaledContent = tf.image.resizeBilinear(
                contentTensor,
                [Math.floor(contentTensor.shape[1] / 2), Math.floor(contentTensor.shape[2] / 2)]
            );
            this.outputImage = tf.variable(scaledContent, true, 'output');
            return true;
        } catch (e) {
            return false;
        }
    }

    _seperated_l2_loss(y_pred, y_true) {
        let sum_axis = [1, 2];
        let _size = y_pred.shape[1] * y_pred.shape[2];
        if (y_pred.shape.length === 4) {
            sum_axis = [1, 2, 3];
            _size = y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3];
        }
        const diff = tf.sub(y_pred, y_true);
        const abs_diff = tf.abs(diff);
        //tf.dispose(diff);
        const square_abs_diff = tf.square(abs_diff);
        //tf.dispose(diff);
        const reduced_abs_diff = tf.sum(square_abs_diff, sum_axis);
        //tf.dispose(square_abs_diff);
        const two = tf.scalar(2.0, 'float32');
        const l2 = tf.div(reduced_abs_diff, two);
        //tf.dispose(reduced_abs_diff);
        const l2t2 = tf.mul(two, l2);
        //tf.dispose([two, l2]);
        const size = tf.scalar(_size, 'float32');
        const sepl2 = tf.div(l2t2, size);
        //tf.dispose([size, l2t2]);
        return sepl2;
    }

    _convert_to_gram_matrix(inputs) {
        const [batch, height, width, filters] = inputs.shape;
        const feats = tf.reshape(inputs, [batch, height * width, filters]);
        const feats_t = tf.transpose(feats, [0, 2, 1]);
        const grams_raw = tf.matMul(feats_t, feats);
        //tf.dispose([feats, feats_t]);
        const size = tf.scalar(height * width * filters, 'float32');
        const gram_matrix = tf.div(grams_raw, size);
        //tf.dispose([grams_raw, size]);
        return gram_matrix
    }

    _styleLoss() {
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
        const styleWeight = tf.scalar(1e0, 'float32');
        const lossOutput = tf.mul(avgLayerLoss, styleWeight);
        //tf.dispose([avgLayerLoss, styleWeight]);
        return lossOutput;
    }

    loss = () => {
        this.computeOutputImage();
        const styleLoss = this._styleLoss();
        const lossScalar = styleLoss.asScalar();
        //tf.dispose(styleLoss);
        console.log(`[${this.iteration + 1}/${this.iterations}] loss: ${lossScalar.dataSync()}`);
        return lossScalar;
    };


    async process(iterations: number) {
        this.iterations = iterations;
        this.iteration = 0;
        // await this.preComputeContent();
        await this.preComputeStyle();
        for (let i = 0; i < iterations; i++) {
            await this.runIteration();
            this.iteration++;
        }
        await saveTensorAsImage(this.outputImage.clone(), this.output)
        // this.finalCleanup();
    }

    async preComputeContent() {
        // const content = tf.fromPixels(document.getElementById('content-canvas'));
        const content = await getImageAsTensor(this.content);
        const vggInput = this.vgg19.prepareInput(content);
        this.contentLayers = await this.vgg19.getLayers(vggInput, 'content');
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
        this.outputImageLayers = this.vgg19.getLayers(vggInput, 'style');
    }

    runIteration() {
        if (this.iteration === Math.floor(this.iterations / 2)) {
            const size = {
                width:Math.floor(this.outputImage.shape[2] * 2),
                height:Math.floor(this.outputImage.shape[1] * 2)
            };
            const newVar = tf.variable(tf.image.resizeBilinear(this.outputImage,[size.height, size.width]));
            tf.dispose(this.outputImage);
            this.outputImage = newVar;
        }
        this.optimizer.minimize(this.loss, false, [this.outputImage]);
    }
}