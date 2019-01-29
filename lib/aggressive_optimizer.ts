import tf from './tensorflow';

function dot(a, b){
    return tf.tidy(()=>{
        const multiplied = tf.mul(a, b);
        return tf.sum(multiplied)
    });
}

export default class AggressiveOptimizer{
    constructor(
        maxIter:number=15000,
        maxFun:number=15000,
        gtol:number=1e-5,
        ftol:number=2.220446049250313e-09,
        maxCor:number=10,
        maxAdjustment:number=5
    ){
        this.maxIter = maxIter;
        this.maxFun = maxFun;
        this.gtol = gtol;
        this.ftol = ftol;
        this.maxCor = maxCor;
        this.maxAdjustment = maxAdjustment;

        this.reset();
    }
    maxIter:number;
    maxFun:number;
    gtol:number;
    ftol:number;
    maxCor:number;
    maxAdjustment:number;

    correction:tf.Tensor;
    oldDirs:tf.Tensor[];
    oldSteps:tf.Tensor[];
    hessian:any;
    gradientOld:tf.Tensor;
    valueOld:tf.Tensor;
    correctionAdjustment:tf.Tensor;

    reset():void{
        this.cleanup();
        this.oldDirs = [];
        this.oldSteps = [];
        this.correction = null;
        this.correctionAdjustment = null;
        this.hessian = tf.scalar(1.0, 'float32');
        this.gradientOld = null;
        this.valueOld = null;
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

    minimize(f: () => tf.Scalar, returnCost?: boolean, varList?: tf.Variable[]): tf.Scalar | null{
        return tf.tidy(()=>{
            let tmp;
            let isFirstIteration;
            const opFunc = (x:tf.Tensor[]) => {
                return tf.tidy(()=>{
                    const g = tf.valueAndGrads(f);
                    const {value, grads} = g(x);
                    return [value, grads[0]]
                });
            };

            let [value, gradient] = opFunc(varList);

            const negOne = tf.scalar(-1.0, 'float32');
            const zeroScalar = tf.scalar(0.0, 'float32');
            const oneScalar = tf.scalar(1.0, "float32");

            if(this.correction === null || this.gradientOld === null){
                isFirstIteration = true;
                this.correction = tf.mul(gradient, negOne);
            } else {
                isFirstIteration = false;
                const y = tf.sub(gradient, this.gradientOld);
                const s = tf.mul(this.correction, this.correctionAdjustment);
                const ys = dot(y, s);

                const ysData = ys.dataSync();
                // todo is ysDaya an array or value?
                if(ysData[0] > 1e-10) {
                    // update memory
                    if (this.oldDirs.length === this.maxCor) {
                        // limit memory to maxCor
                        tmp = this.oldDirs.shift();
                        tf.dispose(tmp);
                        tmp = this.oldSteps.shift();
                        tf.dispose(tmp)
                    }
                    tf.keep(s);
                    tf.keep(y);
                    this.oldDirs.push(s);
                    this.oldSteps.push(y);
                    let hessian =  tf.div(ys, dot(y, y));
                    tf.dispose(this.hessian);
                    tf.keep(hessian);
                    this.hessian = hessian;
                }

                const k = this.oldDirs.length;

                let ro = [];
                for(let i = 0; i < this.maxCor; i++){
                    ro.push(zeroScalar);
                }

                for(let i = 0; i < k; i++){
                    ro[i] = tf.tidy(()=>{
                        return tf.div(oneScalar, dot(this.oldSteps[i], this.oldDirs[i]));
                    });
                }

                let al = [];
                for(let i = 0; i < this.maxCor; i++){
                    al.push(zeroScalar);
                }

                let q = tf.mul(gradient, negOne);
                for(let i = k - 1; i >= 0; i--){
                    al[i] = tf.tidy(()=>{
                        return tf.mul(dot(this.oldDirs[i], q), ro[i]);
                    });
                    tmp = tf.tidy(()=>{
                        return tf.sub(q, tf.mul(al[i], this.oldSteps[i]))
                    });
                    tf.dispose(q);
                    q = tmp;
                }

                // multiply by hessian
                let newCorrection = tf.mul(q, this.hessian);
                for(let i = 0; i < k; i++){
                    tmp = tf.tidy(()=>{
                        const be_i = tf.mul(dot(this.oldSteps[i], newCorrection), ro[i]);
                        const addToCorrection = tf.mul(tf.sub(al[i], be_i), this.oldDirs[i]);
                        return tf.add(newCorrection, addToCorrection);
                    });
                    tf.dispose(newCorrection);
                    newCorrection = tmp;
                }
                tf.dispose(this.correction);
                this.correction = newCorrection;
            }

            // compute step length

            // directional derivative
            const gtd = dot(gradient, this.correction);

            // check that progress can be made along that direction
            const gtdData = gtd.dataSync();
            if(gtdData[0] > -this.ftol){
                console.log("Can not make progress along direction.")
            }

            const newVal = tf.tidy(()=>{
                const meanCorrection = tf.mean(tf.abs(this.correction));
                const absValue = tf.abs(value);
                const correctionBase = tf.log1p(absValue);
                this.correctionAdjustment = tf.keep(tf.div(correctionBase, meanCorrection));

                let adjustment;
                if(isFirstIteration){
                    adjustment = tf.mul(this.correctionAdjustment, this.correction)
                } else {
                    adjustment = tf.mul(this.correction, tf.scalar(5.0, "float32"))
                }

                adjustment = tf.clipByValue(adjustment, -this.maxAdjustment, this.maxAdjustment);

                return tf.add(varList[0], adjustment)
            });
            varList[0].assign(newVal);

            if(!isFirstIteration){
                const valueChange = tf.tidy(()=>{
                    return tf.sum(tf.abs(tf.sub(value, this.valueOld)))
                }).dataSync();
                if(valueChange[0] < this.ftol){
                    console.log('Value change is less than ftol. Giving a bump');
                    const bumpVal = tf.tidy(()=>{
                        const multiplier = tf.scalar(0.9, "float32");
                        return tf.mul(varList[0], multiplier)
                    });
                    varList[0].assign(bumpVal);
                }
            }

            tf.dispose(this.gradientOld);
            tf.dispose(this.valueOld);
            this.gradientOld = gradient;
            this.valueOld = value;
            tf.keep(this.correction);
            tf.keep(this.correctionAdjustment);
            tf.keep(this.valueOld);
            tf.keep(this.gradientOld);
            return null;
        })
    }

}