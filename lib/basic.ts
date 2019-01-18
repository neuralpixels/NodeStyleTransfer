

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

export function exponent(num:any, decimals:number, exponentWidth:number=2) {
    const raw = `${Number.parseFloat(num).toExponential(decimals)}`;
    let splitter = 'e';
    if(raw.includes('e-')){
        splitter = 'e-';
    } else if (raw.includes('e+')) {
        splitter = 'e+';
    }
    const splitArr = raw.split(splitter);
    while(splitArr[1].length < exponentWidth){
        splitArr[1] = `0${splitArr[1]}`;
    }
    return splitArr.join(splitter);
}

export function rjust(text:number|string, width:number=1, spaceChar:string=' '){
    let str = `${text}`;
    if(spaceChar.length !== 1){
        throw('rjust spaceChar needs to be a single character')
    }
    while(str.length < width){
        str = `${spaceChar}${str}`;
    }
    return str;
}

export function ljust(text:number|string, width:number=1, spaceChar:string=' '){
    let str = `${text}`;
    if(spaceChar.length !== 1){
        throw('ljust spaceChar needs to be a single character')
    }
    while(str.length < width){
        str = `${str}${spaceChar}`;
    }
    return str;
}