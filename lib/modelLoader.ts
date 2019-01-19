import tf from './tensorflow';
import {valueMap} from './basic';
import * as io from './io';

export interface IVariables {
    [name: string]: any
}

function dequantize(values, shape,  scale, minValue){
    const tmp = tf.tensor(values, shape, 'int32');
    const quantizedVal = tf.cast(tmp, 'float32');
    tf.dispose(tmp);
    const _scale = tf.scalar(scale, 'float32');
    const _minValue = tf.scalar(minValue, 'float32');

    const scaled = tf.mul(quantizedVal, _scale);
    tf.dispose([quantizedVal, _scale]);
    const added = tf.add(scaled, _minValue);
    tf.dispose([scaled, _minValue]);
    return added;
}

function loadSingleVariables(name, path, shape, variablesObj, dtype, quantization, done) {
    return new Promise((resolve, reject) => {
        io.get(path, true).then((data) => {
            // console.log(name, response);
            const rawData = data;
            if(quantization){
                if (dtype === 'uint8') {
                    const values = new Uint8Array(rawData);
                    variablesObj[name] = dequantize(values, shape, quantization.scale, quantization.min_value);
                } else if (dtype === 'uint16') {
                    const values = new Uint16Array(rawData);
                    variablesObj[name] = dequantize(values, shape, quantization.scale, quantization.min_value);
                }
            } else {
                if (dtype === 'float32') {
                    const values = new Float32Array(rawData);
                    variablesObj[name] = tf.tensor(values, shape);
                } else if (dtype === 'uint8') {
                    const values = new Uint8Array(rawData);
                    const tmp = tf.tensor(values, shape);
                    variablesObj[name] = tf.cast(tmp, 'float32');
                    tf.dispose(tmp);
                }
            }
            done();
            resolve();
        }).catch((error) => {
            console.log(error);
            done();
            reject(error)
        });
    });
}

export function loadRemoteVariablesPromise(modelPath:string):Promise<IVariables> {
    return new Promise((resolve, reject) => {
        loadRemoteVariables(modelPath, (err, data) => {
            if(err){
                reject(err)
            } else {
                resolve(data)
            }
        });
    });
}

export function loadRemoteVariables(modelPath:string, callback) {
    const variablesPath = `${modelPath}`;
    const manifestPath = `${modelPath}/manifest.json`;

    // first get the manifest
    io.get(manifestPath)
        .then((data) => {
            const manifest = data;
            let returnVariables = {};
            let promisesArr = [];

            let numVarsLoaded = 0;
            let numTotalVars;
            let weightObj;
            if ('weights' in manifest){
                // new manifest format
                weightObj = manifest.weights;
                numTotalVars = Object.keys(manifest.weights).length;
            } else {
                // old manifest format
                weightObj = manifest;
                numTotalVars = Object.keys(manifest).length;
            }

            const varDownloaded = () => {
                numVarsLoaded++;
                const percent = valueMap(numVarsLoaded, 0, numTotalVars, 0, 100);

            };
            // build the promises
            for (let varName in weightObj) {
                const weightPath = `${variablesPath}/${weightObj[varName].filename}`;
                const weightShape = weightObj[varName].shape;
                const weightDtype = 'dtype' in weightObj[varName] ? weightObj[varName].dtype : 'float32';
                const weightQuantization = 'quantization' in weightObj[varName] ? weightObj[varName].quantization : null;
                promisesArr.push(
                    loadSingleVariables(
                        varName,
                        weightPath,
                        weightShape,
                        returnVariables,
                        weightDtype,
                        weightQuantization,
                        varDownloaded
                    )
                )
            }

            // wait for all promises to resolve
            Promise.all(promisesArr).then(() => {
                callback(null, returnVariables);
            }).catch(error => {
                callback(error, null);
            });
        })
        .catch((error) => {
            // handle error
            console.log('/currentModels.json err', error);
            callback(error, null);
        });
}
