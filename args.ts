import * as argparse from "argparse";

const parser = new argparse.ArgumentParser({
    description: 'Node Style Transfer',
    addHelp: true
});
parser.addArgument('--content', {
    type: 'string',
    defaultValue: './test/lenna.jpg',
    help: 'Path to content img'
});
parser.addArgument('--style', {
    type: 'string',
    defaultValue: './test/paint01.jpg',
    help: 'Path to style img'
});
parser.addArgument('--output', {
    type: 'string',
    defaultValue: './test/lenna_paint01.jpg',
    help: 'Path to save output'
});
parser.addArgument('--gpu', {
    type: 'int',
    defaultValue: 0,
    help: 'GPU to use'
});
const args = parser.parseArgs();
process.env['TF_CPP_MIN_LOG_LEVEL'] = '2';
process.env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
process.env["CUDA_VISIBLE_DEVICES"] = `${args.gpu}`;

export default args