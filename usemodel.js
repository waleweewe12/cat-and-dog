const path = require('path')
const modelpath  = path.join(__dirname,'models')
const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')

const usemodel=async()=>{
    /*load test data */
    /*let data = fs.readFileSync('./dogtest2.jpg')
    let testdata = tf.node.decodeImage(data)

    let testdata_Array = []
    testdata_Array.push(testdata)
    data = fs.readFileSync('./cattest2.jpg')
    testdata = tf.node.decodeImage(data)
    testdata_Array.push(testdata)*/
    let testdata = []
    let alldata = fs.readdirSync(path.join(__dirname,'alldata')) 
    alldata.forEach(filename=>{
        let data = fs.readFileSync(path.join(__dirname,'alldata')+"\\"+filename)
        data = tf.node.decodeImage(data)
        testdata.push(data) 
    })
    /*divide 255 */
    const b = tf.scalar(255)
    let evaluate = testdata
    testdata = tf.stack(testdata).div(b)
    /*load and train model */
    const model = await tf.loadLayersModel('file://'+modelpath+"//"+'model.json')
    let result = await model.predict(testdata)
    const ypred = result.arraySync();

    ypred.forEach(showresult2)

    function showresult2(item,index){
        console.log(index+1+" "+item[0]+" "+item[1])
    }
    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics:['accuracy']
    })
    let y_test = []
    for(let i=0;i<22;i++)
    {
        if(i<11)
        {
            y_test.push([0,1])
        }else{
            y_test.push([1,0])
        }
    }
    y_test = tf.tensor2d(y_test)
    let test = model.evaluate(testdata,y_test)
    test.forEach(data=>{
        data.print()
    })
}

usemodel()
