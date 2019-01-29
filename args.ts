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

export default args