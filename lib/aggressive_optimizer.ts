import tf from './tensorflow';

function dot(a, b){
    return tf.tidy(()=>{
        const multiplied = tf.mul(a, b);
        return tf.sum(multiplied)
    });
}

export default class AggressiveOptimizer extends tf.Optimizer{
    static className = 'AggressiveOptimizer';
    public maxIter:number;
    maxFun:number;
    gtol:number;
    ftol:number;
    maxCor:number;
    maxAdjustment:number;

    correction:{[name:string]:tf.Tensor};
    oldDirs:{[name:string]:tf.Tensor[]};
    oldSteps:{[name:string]:tf.Tensor[]};
    hessian:{[name:string]:any};
    gradientOld:{[name:string]:tf.Tensor};
    valueOld:{[name:string]:tf.Tensor};
    correctionAdjustment:{[name:string]:tf.Tensor};

    constructor(
        maxIter:number=15000,
        maxFun:number=15000,
        gtol:number=1e-5,
        ftol:number=2.220446049250313e-09,
        maxCor:number=10,
        maxAdjustment:number=5
    ){
        super();
        this.maxIter = maxIter;
        this.maxFun = maxFun;
        this.gtol = gtol;
        this.ftol = ftol;
        this.maxCor = maxCor;
        this.maxAdjustment = maxAdjustment;

        this.reset();
    }

    reset():void{
        this.cleanup();
        this.oldDirs = {};
        this.oldSteps = {};
        this.correction = {};
        this.correctionAdjustment = {};
        this.hessian = {};
        this.gradientOld = {};
        this.valueOld = {};
    }
    cleanup(){
        try{
            tf.dispose([
                this.oldDirs,
                this.oldSteps,
                this.correction,
                this.correctionAdjustment,
                this.hessian,
                this.gradientOld,
                this.valueOld
            ])
        } catch (e){
            console.log(e)
        }
    }
    applyGradients(variableGradients: tf.NamedTensorMap, loss=null): void{
        for(const variableName in variableGradients) {
            tf.tidy(()=> {
                const variable = tf.ENV.engine.registeredVariables[variableName];
                const gradient = variableGradients[variableName];
                // const value = tf.mean(gradient); // todo we really need to use loss value here
                let tmp;
                let isFirstIteration;
                const negOne = tf.scalar(-1.0, 'float32');
                const zeroScalar = tf.scalar(0.0, 'float32');
                const oneScalar = tf.scalar(1.0, "float32");
                const value = loss.clone();

                if (!(variableName in this.correction) || !(variableName in this.gradientOld)) {
                    isFirstIteration = true;
                    this.correction[variableName] = tf.mul(gradient, negOne);
                    this.hessian[variableName] = tf.scalar(1.0, 'float32');
                    tf.keep(this.hessian[variableName]);
                    this.oldDirs[variableName] = [];
                    this.oldSteps[variableName] = [];

                } else {
                    isFirstIteration = false;
                    const y = tf.sub(gradient, this.gradientOld[variableName]);
                    const s = tf.mul(this.correction[variableName], this.correctionAdjustment[variableName]);
                    const ys = dot(y, s);

                    const ysData = ys.dataSync();
                    // todo is ysDaya an array or value?
                    if (ysData[0] > 1e-10) {
                        // update memory
                        if (this.oldDirs[variableName].length === this.maxCor) {
                            // limit memory to maxCor
                            tmp = this.oldDirs[variableName].shift();
                            tf.dispose(tmp);
                            tmp = this.oldSteps[variableName].shift();
                            tf.dispose(tmp)
                        }
                        tf.keep(s);
                        tf.keep(y);
                        this.oldDirs[variableName].push(s);
                        this.oldSteps[variableName].push(y);
                        let hessian = tf.div(ys, dot(y, y));
                        tf.dispose(this.hessian[variableName]);
                        this.hessian[variableName] = hessian;
                        tf.keep(this.hessian[variableName]);
                    }

                    const k = this.oldDirs[variableName].length;

                    let ro = [];
                    for (let i = 0; i < this.maxCor; i++) {
                        ro.push(zeroScalar);
                    }

                    for (let i = 0; i < k; i++) {
                        ro[i] = tf.tidy(() => {
                            return tf.div(oneScalar, dot(this.oldSteps[variableName][i], this.oldDirs[variableName][i]));
                        });
                    }

                    let al = [];
                    for (let i = 0; i < this.maxCor; i++) {
                        al.push(zeroScalar);
                    }

                    let q = tf.mul(gradient, negOne);
                    for (let i = k - 1; i >= 0; i--) {
                        al[i] = tf.tidy(() => {
                            return tf.mul(dot(this.oldDirs[variableName][i], q), ro[i]);
                        });
                        tmp = tf.tidy(() => {
                            return tf.sub(q, tf.mul(al[i], this.oldSteps[variableName][i]))
                        });
                        tf.dispose(q);
                        q = tmp;
                    }

                    // multiply by hessian
                    let newCorrection = tf.mul(q, this.hessian[variableName]);
                    for (let i = 0; i < k; i++) {
                        tmp = tf.tidy(() => {
                            const be_i = tf.mul(dot(this.oldSteps[variableName][i], newCorrection), ro[i]);
                            const addToCorrection = tf.mul(tf.sub(al[i], be_i), this.oldDirs[variableName][i]);
                            return tf.add(newCorrection, addToCorrection);
                        });
                        tf.dispose(newCorrection);
                        newCorrection = tmp;
                    }
                    tf.dispose(this.correction[variableName]);
                    this.correction[variableName] = newCorrection;
                }

                // compute step length

                // directional derivative
                const gtd = dot(gradient, this.correction[variableName]);

                // check that progress can be made along that direction
                const gtdData = gtd.dataSync();
                if (gtdData[0] > -this.ftol) {
                    console.log("Can not make progress along direction.")
                }

                const newVal = tf.tidy(() => {
                    const meanCorrection = tf.mean(tf.abs(this.correction[variableName]));
                    const absValue = tf.abs(value);
                    const correctionBase = tf.log1p(absValue);
                    const correctionAdjustment = tf.div(correctionBase, meanCorrection);
                    tf.dispose(this.correctionAdjustment[variableName]);
                    this.correctionAdjustment[variableName] = correctionAdjustment;
                    tf.keep(this.correctionAdjustment[variableName]);

                    let adjustment;
                    if (isFirstIteration) {
                        adjustment = tf.mul(this.correctionAdjustment[variableName], this.correction[variableName])
                    } else {
                        adjustment = tf.mul(this.correction[variableName], tf.scalar(5.0, "float32"))
                    }

                    adjustment = tf.clipByValue(adjustment, -this.maxAdjustment, this.maxAdjustment);

                    return tf.add(variable, adjustment)
                });
                variable.assign(newVal);

                if (!isFirstIteration) {
                    const valueChange = tf.tidy(() => {
                        return tf.sum(tf.abs(tf.sub(value, this.valueOld[variableName])))
                    }).dataSync();
                    if (valueChange[0] < this.ftol) {
                        console.log('Value change is less than ftol. Giving a bump');
                        const bumpVal = tf.tidy(() => {
                            const multiplier = tf.scalar(0.9, "float32");
                            return tf.mul(variable, multiplier)
                        });
                        variable.assign(bumpVal);
                    }
                }

                tf.dispose(this.gradientOld[variableName]);
                tf.dispose(this.valueOld[variableName]);
                this.gradientOld[variableName] = gradient;
                this.valueOld[variableName] = value;
                tf.keep(this.correction[variableName]);
                tf.keep(this.correctionAdjustment[variableName]);
                tf.keep(this.valueOld[variableName]);
                tf.keep(this.gradientOld[variableName]);
                return null;
            })
        }
        return null;
    }

    getConfig() {
        return {
        };
    }
}