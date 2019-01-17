

export function valueMap( x:number,  in_min:number,  in_max:number,  out_min:number,  out_max:number):number{
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

export function wait(ms:number):Promise<null>{
    return new Promise<null>(resolve =>{
        setTimeout(() => {
            resolve();
        }, ms)
    })
}
