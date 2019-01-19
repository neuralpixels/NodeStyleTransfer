import {isNode} from './env';
import axios from 'axios';


export function get(filePath: string, binary: boolean = false): Promise<any> {
    return new Promise((resolve, reject) => {
        if (isNode) {
            const fs = require('fs');
            try{
                const data = fs.readFileSync(filePath);
                if(binary){
                    const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
                    resolve(arrayBuffer);
                } else {
                    const extension = filePath.split('.').pop();
                    if(extension === 'json'){
                        try{
                            const json = JSON.parse(data);
                            resolve(json);
                        } catch(e){
                            resolve(data);
                        }
                    }
                }
            } catch (e) {
                reject(e);
            }

        } else {
            // for web based
            let config = {};
            if (binary) {
                config = {
                    responseType: 'arraybuffer'
                }
            }
            axios.get(filePath, config).then((response) => {
                if (response.status === 200) {
                    resolve(response.data)
                } else {
                    reject(`Got status ${response.status} from server`)
                }
            }).catch((error) => {
                reject(error);
            })
        }

    });
}