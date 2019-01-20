import tf from './tensorflow'
import {wait} from "./basic";
import {biasAdd} from './tfextra';
import {IVariables} from './modelLoader'

async function renderBreak(delay = 1) {
    // call any triggers waiting to update
    await tf.nextFrame();
    if (delay > 0) {
        await wait(delay);
    }
}

export interface Vgg19LayerWeights {
    [layer: string]: {
        [lossType:string]:number
    }
}

class VGG19{
    constructor(variables:IVariables){
        this.variables = variables;

        this.vgg19_layers = [
            'conv1_1', 'conv1_2', 'pool1',
            'conv2_1', 'conv2_2', 'pool2',
            'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'pool3',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4',
            'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5'
        ];
        // this.vgg19_content_layers = [
        //     'conv4_2', 'conv5_2'
        // ];
        this.vgg19_content_layers = [
            'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'
        ];
        this.vgg19_style_layers = [
            'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'
        ];
        this.vgg19_layer_weights = {
            "conv1_1": {
                "content": 0.0003927100042346865,
                "style": 0.27844879031181335
            },
            "conv1_2": {
                "content": 2.99037346849218e-05,
                "style": 0.0004943962558172643
            },
            "conv2_1": {
                "content": 2.0568952095345594e-05,
                "style": 0.0009304438135586679
            },
            "conv2_2": {
                "content": 1.073586827260442e-05,
                "style": 0.00040253016049973667
            },
            "conv3_1": {
                "content": 1.0999920050380751e-05,
                "style": 0.0001156232028733939
            },
            "conv3_2": {
                "content": 1.0808796105266083e-05,
                "style": 7.009495311649516e-05
            },
            "conv3_3": {
                "content": 4.947870365867857e-06,
                "style": 7.687774996156804e-06
            },
            "conv3_4": {
                "content": 1.2470403589759371e-06,
                "style": 8.033587732825254e-07
            },
            "conv4_1": {
                "content": 1.4441507119045127e-06,
                "style": 5.199814836487349e-07
            },
            "conv4_2": {
                "content": 2.3558966404380044e-06,
                "style": 2.2772749161958927e-06
            },
            "conv4_3": {
                "content": 5.842243808729108e-06,
                "style": 2.7995649361400865e-05
            },
            "conv4_4": {
                "content": 3.0219671316444874e-05,
                "style": 0.001985269133001566
            },
            "conv5_1": {
                "content": 6.438765558414161e-05,
                "style": 0.000784530770033598
            },
            "conv5_2": {
                "content": 0.00033032899955287576,
                "style": 0.018374426290392876
            },
            "conv5_3": {
                "content": 0.0016348531935364008,
                "style": 0.42564332485198975
            },
            "conv5_4": {
                "content": 0.02764303795993328,
                "style": 95.27446746826172
            }
        }

    }
    public variables:IVariables;
    public vgg19_layers:Array<string>;
    public vgg19_content_layers:Array<string>;
    public vgg19_style_layers:Array<string>;
    public vgg19_layer_weights:Vgg19LayerWeights;

    _pool(inputs) {
        return tf.tidy(()=>{
            return tf.maxPool(inputs, [2, 2], [2, 2], 'same');
        });
    }
    _conv(inputs, name){
        return tf.tidy(()=>{
            const conv = tf.conv2d(inputs, this.variables[`${name}_kernel`], [1, 1], 'same');
            const bias = biasAdd(conv, this.variables[`${name}_bias`]);
            return tf.relu(bias);
        });
    }

    prepareInput(inputs){
        return tf.tidy(()=>{
            let floatInputs = inputs;
            // convert input to float32 if not already
            if(floatInputs.dtype === 'int32') {
                floatInputs = tf.cast(inputs, 'float32');
            }
            if(floatInputs.shape.length === 3){
                const expanded = tf.expandDims(floatInputs, 0);
                floatInputs = expanded;
            }
            // transpose rgb => bgr and adjust to mean pixel
            const chAxis = floatInputs.shape.length - 1;
            const [r, g, b] = tf.split(floatInputs, 3, chAxis);
            const vggMean = [tf.scalar(103.939), tf.scalar(116.779), tf.scalar(123.68)];
            const scaledArr = [
                tf.sub(b, vggMean[0]),
                tf.sub(g, vggMean[1]),
                tf.sub(r, vggMean[2])
            ];
            return tf.concat(scaledArr, chAxis);
        });
    }

    getLayers(inputs, type='content'){
        let returnLayers = {};
        let layersToGet;
        if(type === 'content'){
            layersToGet = this.vgg19_content_layers;
        } else if(type === 'style') {
            layersToGet = this.vgg19_style_layers;
        } else {
            // get both
            layersToGet = this.vgg19_style_layers.concat(this.vgg19_content_layers);
        }
        layersToGet.sort();
        let lastLayer = layersToGet[layersToGet.length - 1];
        let next = inputs;
        for(let i = 0; i < this.vgg19_layers.length; i++){
            let layer = this.vgg19_layers[i];
            if(layer.includes('conv')){
                // do convolution
                next = this._conv(next, layer);
                // clone it to return object if it is a layer to get
                if(layersToGet.includes(layer)){
                    returnLayers[layer] = next.clone();
                }
            } else if(layer.includes('pool')) {
                // do max pool
                next = this._pool(next);
            }

            // stop if this was last layer
            if(layer === lastLayer){
                break;
            }
        }
        //tf.dispose(next);
        return returnLayers;
    }
}


export default VGG19