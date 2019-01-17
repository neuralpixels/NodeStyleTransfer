import args from './args'
import {StyleTransfer} from './lib/styleTransfer'

async function run(
    parameters: { content: string, style: string, output: string, iterations?:number}
) {
    let {content, style, output, iterations=1000} = parameters;
    const styleTransfer = new StyleTransfer({
        content:content,
        style:style,
        output:output
    });
    console.log('Initializing');
    await styleTransfer.initialize();
    await styleTransfer.process(iterations)
}

if (typeof require != 'undefined' && require.main==module) {
    run({
        content:args.content,
        style:args.style,
        output:args.output
    });
}