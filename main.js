

let model;

const modelURL = 'http://localhost:5000/model';

const formInput1 = document.getElementById("formInput1");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
//const numberOfFiles = document.getElementById("number-of-files");
//const fileInput = document.getElementById('file');


const predict = async (modelURL) => {
    if (!model) model = await tf.loadLayersModel(modelURL);
    //    const files = fileInput.files;

//    [...files].map(async (img) => {
    const formInput1 = document.getElementById('formInput1').value;
    const data = new FormData();
    data.append('formInput1', formInput1);

    const processedImage = await fetch("http://localhost:5000/api/prepare", {
        method: 'POST',
        mode: 'cors',
        body: data
    }).then(response => {
            return response.json();
        }).then(result => {
            console.log('resposne: ', result['ndarray'][0]);
            return tf.tensor(result['ndarray']);
        });
    // shape has to be the same as it was for training of the model

    let prediction = model.predict(processedImage);
//        console.log(${prediction});
    console.log("prediction :- ", prediction);

//        result =
//        result =  (label > .5) ? 'Accepted' : 'Rejected';
//        renderImageLabel(img, label);
//    res = tf.argMax(prediction, 1).dataSync()[0];
//    console.log("Modified prediction", res)
return res;
//    })
};


predictButton.addEventListener("click", () => predict(modelURL));
clearButton.addEventListener("click", () => {
    formInput1.value = "";
    console.log('Clear')
});
