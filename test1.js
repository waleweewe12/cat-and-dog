const tf = require('@tensorflow/tfjs-node')
const path = require('path')
const fs = require('fs')
const sharp = require('sharp')

let trainData = []
const tensorFeature_cat = []
const tensorFeature_dog = []
let tensorFeatures
let AllFeatures = []

const loaddata = ()=>{
    /*load cat image */
    let directoryPath = path.join(__dirname, 'cat');
    let files = fs.readdirSync(directoryPath)
    files.forEach(function (file) {
        let data = fs.readFileSync(directoryPath+"\\"+file)
       /* sharp(data)
            .resize(256, 256)
            .toFile(directoryPath+"\\"+file, (err, info) => { })
        data = fs.readFileSync(directoryPath+"\\"+file) */
        trainData.push(data) 
    });
    
    for(let i=0;i<trainData.length;i++)
    {
        tensorFeature_cat[i] = tf.node.decodeImage(trainData[i])
    }

    trainData = []

    /*load dog image */
    directoryPath = path.join(__dirname, 'dog');
    files = fs.readdirSync(directoryPath)
    files.forEach(function (file) {
        let data = fs.readFileSync(directoryPath+"\\"+file)
        /*sharp(data)
            .resize(256, 256)
            .toFile(directoryPath+"\\"+file, (err, info) => { })
        data = fs.readFileSync(directoryPath+"\\"+file)*/
        trainData.push(data)
    });
    
    for(let i=0;i<trainData.length;i++)
    {
        tensorFeature_dog[i] = tf.node.decodeImage(trainData[i])
    }
    //tensorFeatures = tf.stack([tensorFeature_cat,tensorFeature_dog])
    
    for(let i=0;i<20;i++)
    {
        if(i<10)
        {
            AllFeatures.push(tensorFeature_cat[i])
        }
        else{
            AllFeatures.push(tensorFeature_dog[i-10])
        }
    }

    tensorFeatures = tf.stack(AllFeatures)
    //console.log(tensorFeatures)
}

loaddata()

/*for(let i=0;i<AllFeatures.length;i++)
{
    console.log(AllFeatures[i])
}*/

let labelArray = []
for(let i=0;i<20;i++)
{
    if(i<10)
    {
        labelArray.push([1,0])
    }
    else{
        labelArray.push([0,1])
    }
}
//tf.tensor2d(labelArray).print()

let tensorLabels = tf.tensor2d(labelArray)

const model = tf.sequential();
model.add(
    tf.layers.dense({
        units: 256,
        inputShape: [256,256,3],
        activation: "relu"
    })
)
model.add(
    tf.layers.dense({
        units: 256,
        activation: "relu"
    })
)
model.add(tf.layers.flatten())
model.add(
    tf.layers.dense({
        units: 2, 
        activation: 'softmax'
    })
)

const LEARNING_RATE = 0.005;

model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: "meanSquaredError"
});

model.fit(tensorFeatures, tensorLabels, {
    epochs: 10
});
let modelpath = path.join(__dirname,'models')
model.save('file://'+modelpath)

let data = fs.readFileSync('./dogtest.jpg')
/*sharp(data)
    .resize(256, 256)
    .toFile('./dogtest.jpg', (err, info) => { })
data = fs.readFileSync('./dogtest.jpg')*/


let testdata = tf.node.decodeImage(data)

let testdata_Array = []
testdata_Array.push(testdata)

const showresult = async()=>{
    let result = await model.predict(tf.stack(testdata_Array))
    const ypred = result.flatten().arraySync();

    ypred.forEach(showresult2)

    function showresult2(item,index){
        console.log(index+1+" "+item)
    }
}

//showresult()
