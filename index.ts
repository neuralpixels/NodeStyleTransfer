import args from './args'
process.env['TF_CPP_MIN_LOG_LEVEL'] = '2';
process.env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
process.env["CUDA_VISIBLE_DEVICES"] = `${args.gpu}`;
import {StyleTransfer} from './lib/styleTransfer'
import {getImageAsTensor, saveTensorAsImage} from './lib/image';

async function run(
    parameters: { content: string, style: string, output: string, iterations?:number}
) {
    let {content, style, output, iterations=50} = parameters;
    const styleTransfer = new StyleTransfer({
        content:content,
        style:style,
        output:output
    });
    console.log('Initializing');
    await styleTransfer.initialize();
    const contentTensor = await getImageAsTensor(args.content, true,'float32');
    const outputTensor = await styleTransfer.process(contentTensor, iterations);
    await saveTensorAsImage(outputTensor, output)
}

if (typeof require != 'undefined' && require.main==module) {
    run({
        content:args.content,
        style:args.style,
        output:args.output
    });
}