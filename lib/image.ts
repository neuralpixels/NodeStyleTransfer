import * as Jimp from 'jimp';
import * as tf from '@tensorflow/tfjs-node-gpu';
import * as path from 'path';

export function getImage(src:string):Promise<Jimp>{
    return new Promise((resolve, reject) => {
        Jimp.read(src)
            .then(img => {
                resolve(img);
            })
            .catch(err => {
                reject(err);
            });
    });
}

export function tensorToImage(inputs:tf.Tensor3D):Promise<Jimp>{
    return new Promise((resolve, reject) => {
        const transposed = tf.transpose(inputs, [1, 0, 2]);
        let dtyped = null;
        if(inputs.dtype !== 'int32'){
            dtyped = tf.cast(transposed, 'int32');
            tf.dispose(transposed);
        } else {
            dtyped = transposed;
        }
        tf.dispose(inputs);
        const alpha = tf.fill([transposed.shape[0], transposed.shape[0], 1], 255, "int32");
        const rgba = tf.concat3d([dtyped, alpha], 2);
        tf.dispose([dtyped, alpha]);
        const width = rgba.shape[0];
        const height = rgba.shape[1];
        rgba.data().then(rawBuffer => {
            tf.dispose(rgba);
            const uint8Buffer = new Uint8Array(rawBuffer);
            // Jimp.read(new Buffer(uint8Buffer.buffer)).then(img => {
            new Jimp(width, height, (err, image) => {
                if(err){
                    reject(err);
                } else {
                    image.bitmap.data = new Buffer(uint8Buffer.buffer);
                    image.bitmap.width = width;
                    image.bitmap.height = height;
                    resolve(image);
                }
            })
        }).catch(err => {
            reject(err);
        });
    });
}

export function getImageAsTensor(src:string):Promise<tf.Tensor3D>{
    return new Promise((resolve, reject) => {
        getImage(src).then(jimpImg => {
            const rgba = tf.tensor3d(jimpImg.bitmap.data, [jimpImg.bitmap.width, jimpImg.bitmap.height, 4]);
            const rgb = tf.slice3d(rgba,[0, 0, 0], [jimpImg.bitmap.width, jimpImg.bitmap.height, 3]);
            tf.dispose(rgba);
            const transposed = tf.transpose(rgb, [1, 0, 2]);
            tf.dispose(rgb);
            resolve(transposed);
        }).catch(err => {
            reject(err)
        });
    });
}

export function saveTensorAsImage(inputs:tf.Tensor3D, dest:string, ):Promise<null>{
    return new Promise((resolve, reject) => {
        tensorToImage(inputs).then(jimpImg => {
            jimpImg.write(dest);
            resolve();
        }).catch(err => {
            reject(err)
        });
    });
}

async function main(){
    const imgPath = path.join(__dirname, '../test/lenna.jpg' );
    const savePath = path.join(__dirname, '../test/lenna_save.jpg' );
    const img = await getImageAsTensor(imgPath);
    await saveTensorAsImage(img, savePath);
    console.log('Done');
}

if (typeof require != 'undefined' && require.main==module) {
    main()
}